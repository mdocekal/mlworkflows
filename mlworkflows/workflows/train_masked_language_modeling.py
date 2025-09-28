#!/usr/bin/env python
from typing import Optional

from classconfig import ConfigurableValue, ConfigurableSubclassFactory
from classconfig.validators import AnyValidator, IsNoneValidator, \
    ValueInIntervalFloatValidator, ValueInIntervalIntegerValidator

from mlworkflows.etl.collators import StructMaskedLanguageModelingDataCollator, MaskedLanguageModelingDataCollator, \
    MaskedLanguageModelingFlatteningDataCollator
from mlworkflows.models.masked_language_modeling import MaskedLanguageModelFactory
from mlworkflows.workflows.base import BaseTrainingWorkflow


class MaskedLanguageModelingTrainingWorkflow(BaseTrainingWorkflow):
    model_fct: MaskedLanguageModelFactory = ConfigurableSubclassFactory(
        MaskedLanguageModelFactory,
        desc="Hugging Face model configuration.",
        name="model",
        user_default=MaskedLanguageModelFactory)
    input: Optional[str] = ConfigurableValue(
        desc="Field in the dataset that contains the text. It will be used to create the input for the model when the dataset is not preprocessed.",
        user_default="text")

    struct_batch: bool = ConfigurableValue(
        desc="Whether to use batching into one vector with struct indices. For evaluation set inputs in include_for_metrics.",
        user_default=False,
        voluntary=True
    )
    flattened_batch: bool = ConfigurableValue(
        desc="Whether to use flattened batching into one vector without struct indices.",
        user_default=False,
        voluntary=True
    )
    selection_prob: float = ConfigurableValue(
        "Probability that given token will be selected for masked language modeling.",
        user_default=0.2, voluntary=True,
        validator=ValueInIntervalFloatValidator(min_value=0.0, max_value=1.0))
    masking_prob: float = ConfigurableValue("Probability that selected token will be replaced with MASK token.",
                                                user_default=1.0, voluntary=True,
                                                validator=ValueInIntervalFloatValidator(min_value=0.0, max_value=1.0))
    random_prob: float = ConfigurableValue("Probability that selected token will be replaced with random token.",
                                               user_default=0.0, voluntary=True,
                                               validator=ValueInIntervalFloatValidator(min_value=0.0, max_value=1.0))
    span_selection_prob: float = ConfigurableValue(
        "Probability that given span will be selected for masked language modeling.",
        user_default=0.0, voluntary=True,
        validator=ValueInIntervalFloatValidator(min_value=0.0, max_value=1.0))
    span_sep_token: Optional[str] = ConfigurableValue("ID of token that will be used to separate spans.",
                                                          user_default=None, voluntary=True,
                                                          validator=AnyValidator(
                                                              [IsNoneValidator(), ValueInIntervalIntegerValidator(min_value=0)]))
    warmup_masked_lm_steps: int = ConfigurableValue(
        "Number of warmup steps for masked language modeling. It will linearly increase the selection_prob from 0 to the value of "
        "selection_prob parameter.",
        user_default=0, voluntary=True,
        validator=ValueInIntervalIntegerValidator(min_value=0)
    )
    min_warmup_selection_prob: float = ConfigurableValue(
        "Minimum selection probability during warmup. It will linearly increase the selection_prob from this value to the value of "
        "selection_prob parameter.",
        user_default=0.0, voluntary=True,
        validator=ValueInIntervalFloatValidator(min_value=0.0, max_value=1.0)
    )

    def __post_init__(self):
        self.input_template = None

    def _init_components(self):
        super()._init_components()
        if not self.dataset_fct.preprocessed:
            self.dataset = self.dataset.map(self.tokenize, batched=True, load_from_cache_file=True)

    def _init_tokenizer(self):
        super()._init_tokenizer()
        if not self.tokenizer.is_fast:
            raise ValueError("Tokenizer is not fast. Please use model with fast tokenizer.")

    def _init_model(self):
        self.model = self.model_fct.create()

    def _init_collator(self):
        if self.struct_batch:
            if not hasattr(self.model.config, "levels"):
                raise ValueError("Model config does not have 'levels' attribute. "
                                 "Please set 'struct_batch' to False or use a model with 'levels' attribute.")
            lvls = self.model.config.levels
            self.collator = StructMaskedLanguageModelingDataCollator(
                tokenizer=self.tokenizer,
                levels=lvls,
                selection_prob=self.selection_prob,
                masking_prob=self.masking_prob,
                random_prob=self.random_prob,
                span_selection_prob=self.span_selection_prob,
                span_sep_token=self.span_sep_token,
                warmup_masked_lm_steps=self.warmup_masked_lm_steps,
                min_warmup_selection_prob=self.min_warmup_selection_prob,
                get_steps=lambda: self.trainer.state.global_step,
                ignore=-100,
                make_matrix_compatible=True)
        elif self.flattened_batch:
            self.collator = MaskedLanguageModelingFlatteningDataCollator(
                tokenizer=self.tokenizer,
                selection_prob=self.selection_prob,
                span_selection_prob=self.span_selection_prob,
                masking_prob=self.masking_prob,
                random_prob=self.random_prob,
                span_sep_token=self.span_sep_token,
                warmup_masked_lm_steps=self.warmup_masked_lm_steps,
                min_warmup_selection_prob=self.min_warmup_selection_prob,
                get_steps=lambda: self.trainer.state.global_step,
                ignore=-100
            )
        else:
            self.collator = MaskedLanguageModelingDataCollator(
                tokenizer=self.tokenizer,
                padding=True,
                max_length=self.tokenizer.model_max_length,
                selection_prob=self.selection_prob,
                masking_prob=self.masking_prob,
                random_prob=self.random_prob,
                span_selection_prob=self.span_selection_prob,
                span_sep_token=self.span_sep_token,
                warmup_masked_lm_steps=self.warmup_masked_lm_steps,
                min_warmup_selection_prob=self.min_warmup_selection_prob,
                get_steps=lambda: self.trainer.state.global_step,
                ignore=-100,
                do_not_select_pad=True,
            )

    def batch_create_input_text(self, batch: dict[str, list[str]]) -> list[str]:
        """
        Creates input text for the batch using the provided template.

        :param batch: Batch of data to process.
        :return: List of input texts.
        """

        if self.input_template is None:
            self.input_template = self.input.create()

        res = []
        for i in range(len(next(iter(batch.values())))):
            sample = {key: value[i] for key, value in batch.items()}
            res.append(self.input_template.render(sample))
        return res

    def tokenize(self, batch: dict[str, list]) -> dict[str, list]:
        """
        Processes a batch of data and tokenizes it.

        :param batch: Batch of data to process.
        :return: Dictionary with tokenized inputs.
        """
        input_texts = self.batch_create_input_text(batch)
        input_ids = self.tokenizer(input_texts, truncation=True, return_attention_mask=False)

        batch["input_ids"] = input_ids["input_ids"]

        return batch


RUN = MaskedLanguageModelingTrainingWorkflow

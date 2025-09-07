#!/usr/bin/env python
import jinja2
import torch
from classconfig import ConfigurableValue, ConfigurableSubclassFactory
from transformers import DataCollatorWithPadding, EvalPrediction

from mlworkflows import models as mlworkflows_models
from mlworkflows.etl.collators import StructDataCollatorForTokenClassification
from mlworkflows.utils.constants import HF_TRAINER_PAD_INDEX
from mlworkflows.workflows.base import BaseTrainingWorkflow


class SequenceClassificationTrainingWorkflow(BaseTrainingWorkflow):
    model_fct: mlworkflows_models.SequenceClassificationModelFactory = ConfigurableSubclassFactory(
        mlworkflows_models.SequenceClassificationModelFactory,
        desc="Hugging Face model configuration.",
        name="model",
        user_default=mlworkflows_models.SequenceClassificationModelFactory)
    input_template: str = ConfigurableValue(
        desc="Jinja2 template for the model input. You can use any field from the dataset.",
        user_default="{{text}}")
    labels_field: str = ConfigurableValue(
        desc="Field in the dataset that contains the label. It will be used to create the target for the model.",
        user_default="labels")
    truncation: bool = ConfigurableValue(
        desc="Whether to truncate the input sequences to the maximum length supported by the model.",
        user_default=True
    )
    indexed_batch: bool = ConfigurableValue(
        desc="Whether to use batching into one vector with batch indices. ",
        user_default=False,
        voluntary=True
    )

    def __post_init__(self):
        if self.compute_metrics is not None:
            self.compute_metrics.transform = self.transform_eval_pred
        self.jinja_env = jinja2.Environment()
        self.jinja_input_template = self.jinja_env.from_string(self.input_template)

    def _init_components(self):
        super()._init_components()
        self.dataset = self.dataset.map(self.tokenize_and_prepare_labels, batched=True, load_from_cache_file=True)

    def _init_tokenizer(self):
        super()._init_tokenizer()
        if not self.tokenizer.is_fast:
            raise ValueError("Tokenizer is not fast. Please use model with fast tokenizer.")

    def _init_model(self):
        self.model = self.model_fct.create(
            dataset=self.dataset[self.dataset_fct.train_split],
            labels_feature=self.labels_field
        )

    def _init_collator(self):
        if self.indexed_batch:
            if not hasattr(self.model.config, "levels"):
                raise ValueError("Model configuration does not have 'levels' defined. "
                                 "Please use a model that supports levels for indexed batching.")

            self.collator = StructDataCollatorForTokenClassification(
                pursuit_minimal_context=False,
                levels=self.model.config.levels,
                make_matrix_compatible=True
            )
        else:
            self.collator = DataCollatorWithPadding(self.tokenizer)

    def tokenize_and_prepare_labels(self, examples: dict) -> dict:
        """
        Tokenizes dataset and prepare labels for sequence classification.

        :param examples: Examples to tokenize.
        :return: Tokenized examples with aligned labels.
        :raises ValueError: If tokenization and label alignment fails.
        """
        if not examples:
            return {}

        # check if batch or single example
        single = False
        model_inputs = []
        if isinstance(examples[self.labels_field], str):
            single = True
            model_inputs = self.jinja_input_template.render(examples)
            if len(model_inputs) == 0:
                raise ValueError("Empty model input")
        else:
            for sample_fields in zip(*examples.values()):
                i = self.jinja_input_template.render(dict(zip(examples.keys(), sample_fields)))
                if len(i) == 0:
                    raise ValueError("Empty model input")
                model_inputs.append(i)

        tokenized_inputs = self.tokenizer(
            model_inputs,
            truncation=self.truncation,
            return_offsets_mapping=True
        )

        tokenized_inputs["labels"] = examples[self.labels_field]
        if single:
            tokenized_inputs = {k: v[0] for k, v in tokenized_inputs.items()}
        return tokenized_inputs

    def transform_eval_pred(self, eval_preds: EvalPrediction):
        """
        Transforms the evaluation predictions that are in form of scores to predictions.

        :param eval_preds: Evaluation predictions.
        :return:
        """
        predictions, labels = eval_preds.predictions, eval_preds.label_ids
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()

        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        if self.indexed_batch:
            # we need to remove the artificial batch dimension
            predictions = predictions.reshape(-1, predictions.shape[-1])
            labels = labels.reshape(-1)

        predictions = predictions.argmax(axis=-1)

        new_predictions = []
        new_labels = []
        for p, l in zip(predictions, labels):
            if l == HF_TRAINER_PAD_INDEX:
                break

            new_predictions.append(p)
            new_labels.append(l)

        eval_preds.predictions, eval_preds.label_ids = new_predictions, new_labels


RUN = SequenceClassificationTrainingWorkflow

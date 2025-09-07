#!/usr/bin/env python
from typing import Optional

from classconfig import ConfigurableValue, ConfigurableSubclassFactory, ConfigurableMixin, ConfigurableFactory, \
    RelativePathTransformer, DelayedFactory
from classconfig.transforms import CPUWorkersTransformer
from classconfig.validators import MinValueIntegerValidator, AnyValidator, IsNoneValidator

from mlworkflows.etl.dataset import DatasetFactory
from mlworkflows.etl.template import Template
from mlworkflows.etl.tokenizer import TokenizerFactory


class TokenizeDatasetWorkflow(ConfigurableMixin):
    dataset_fct: DatasetFactory = ConfigurableSubclassFactory(DatasetFactory,
                                                              desc="Hugging Face dataset for training and validation.",
                                                              name="dataset",
                                                              user_default=DatasetFactory)
    split: str = ConfigurableValue(
        desc="Name of the split to use for tokenization.",
        user_default="train",
    )
    tokenizer_fct: Optional[TokenizerFactory] = ConfigurableSubclassFactory(TokenizerFactory,
                                                                            desc="Hugging Face tokenizer for the model. Leave empty if you wish to initialize it from the model.",
                                                                            name="tokenizer",
                                                                            voluntary=True,
                                                                            user_default=TokenizerFactory)
    input: DelayedFactory[Template] = ConfigurableFactory(
        Template,
        desc="Jinja2 template for the input text that will be tokenized",
        delay_init=True,    # It is delayed because of parallel processing, there was a problem with pickling jinja2 environment
    )
    input_name_in_dataset: str = ConfigurableValue(
        desc="Name of the assembled input in the resulting dataset. Is used when you want to also keep the rendered input text in the dataset.",
        user_default="input_text",
        voluntary=True,
    )

    num_proc: int = ConfigurableValue(
        desc="Number of processes to use for processing the dataset.",
        transform=CPUWorkersTransformer()
    )

    result_path: str = ConfigurableValue(
        desc="Path to save the tokenized dataset.",
        transform=RelativePathTransformer()
    )
    max_shard_size: Optional[int] = ConfigurableValue(
        desc="Maximum size of the shard in bytes. If not provided, the dataset will not be sharded.",
        user_default=None,
        voluntary=True,
        validator=AnyValidator([IsNoneValidator(), MinValueIntegerValidator(min_value=1)])
    )
    columns: list[str] = ConfigurableValue(
        desc="List of columns to keep in the dataset after tokenization. By default, only 'input_ids' is kept.",
        user_default=["input_ids"],
        voluntary=True,
    )

    def __post_init__(self):
        self.tokenizer = self.tokenizer_fct.create()
        self.input_template = None
        self.add_input_ids = "input_ids" in self.columns
        self.add_input_text = self.input_name_in_dataset in self.columns

    def __call__(self):
        dataset = self.dataset_fct.create(split=self.split)
        dataset = dataset.map(
            self.batch_process,
            batched=True,
            num_proc=self.num_proc,
            remove_columns=[c for c in dataset.column_names if c not in self.columns],
        )
        dataset.save_to_disk(self.result_path,
                                max_shard_size=self.max_shard_size)

    def batch_create_input_text(self, batch: dict[str, list[str]]) -> list[str]:
        """
        Creates input text for the batch using the provided template.

        :param batch: Batch of data to process.
        :return: List of input texts.
        """
        res = []
        for i in range(len(next(iter(batch.values())))):
            sample = {key: value[i] for key, value in batch.items()}
            res.append(self.input_template.render(sample))
        return res

    def batch_process(self, batch: dict[str, list]) -> dict[str, list]:
        """
        Processes a batch of data and tokenizes it.

        :param batch: Batch of data to process.
        :return: Dictionary with tokenized inputs.
        """
        if self.input_template is None:
            self.input_template = self.input.create()

        input_texts = self.batch_create_input_text(batch)
        input_ids = self.tokenizer(input_texts, truncation=True, return_attention_mask=False)

        if self.add_input_ids:
            batch["input_ids"] = input_ids["input_ids"]
        if self.add_input_text:
            batch[self.input_name_in_dataset] = input_texts

        return batch


RUN = TokenizeDatasetWorkflow

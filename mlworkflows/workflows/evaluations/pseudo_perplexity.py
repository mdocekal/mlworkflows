#!/usr/bin/env python
import json
import math
from collections import defaultdict
from typing import Optional, Sequence

import torch
from accelerate import Accelerator
from classconfig import ConfigurableValue, ConfigurableSubclassFactory, ConfigurableMixin, ConfigurableFactory, \
    DelayedFactory, RelativePathTransformer
from classconfig.validators import MinValueIntegerValidator, AnyValidator, IsNoneValidator, \
    ValueInIntervalFloatValidator, IntegerValidator, BoolValidator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer, \
    BatchEncoding

from mlworkflows.etl.collators import StructDataCollatorForTokenClassification
from mlworkflows.etl.dataset import DatasetFactory
from mlworkflows.etl.template import Template
from mlworkflows.etl.tokenizer import TokenizerFactory
from mlworkflows.models.masked_language_modeling import MaskedLanguageModelFactory


class DataCollator:
    """
    Custom data collator
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, attention_mask: bool = True, ignore_index: int = -100):
        self.tokenizer = tokenizer
        self.attention_mask = attention_mask
        self.ignore_index = ignore_index

    @staticmethod
    def list_of_dicts_to_dict_of_lists(features: list[dict]) -> dict:
        """
        Converts a list of dictionaries to a dictionary of lists.
        :param features: list of dictionaries
        :return: dictionary of lists
        """
        res = defaultdict(list)
        for sample in features:
            for key, value in sample.items():
                res[key].append(value)
        return res

    def __call__(self, features: list[dict]) -> BatchEncoding:
        features = self.list_of_dicts_to_dict_of_lists(features)
        to_pad = {
            "input_ids": features["input_ids"],
        }

        batch = self.tokenizer.pad(to_pad, return_tensors="pt", return_attention_mask=self.attention_mask)

        batch["sample_id"] = torch.tensor(features["sample_id"], dtype=torch.long)
        batch["labels"] = pad_sequence(
            [torch.tensor(l, dtype=torch.long) for l in features["labels"]],
            batch_first=True,
            padding_value=self.ignore_index
        )

        return batch


class PseudoPerplexityWorkflow(ConfigurableMixin):
    dataset_fct: DatasetFactory = ConfigurableSubclassFactory(DatasetFactory,
                                                              desc="Hugging Face dataset for training and validation.",
                                                              name="dataset",
                                                              user_default=DatasetFactory)
    split: str = ConfigurableValue(
        desc="Name of the split to use for calculating pseudo-perplexity.",
        user_default="train",
    )
    model_fct: MaskedLanguageModelFactory = ConfigurableSubclassFactory(
        MaskedLanguageModelFactory,
        desc="Hugging Face model configuration.",
        name="model",
        user_default=MaskedLanguageModelFactory)
    tokenizer_fct: Optional[TokenizerFactory] = ConfigurableSubclassFactory(TokenizerFactory,
                                                                            desc="Hugging Face tokenizer for the model. Leave empty if you wish to initialize it from the model.",
                                                                            name="tokenizer",
                                                                            voluntary=True)
    input: DelayedFactory[Template] = ConfigurableFactory(
        Template,
        desc="Jinja2 template for the input text that will be tokenized",
        delay_init=True,
        # It is delayed because of parallel processing, there was a problem with pickling jinja2 environment
    )
    random_seed: Optional[int] = ConfigurableValue(
        desc="Random seed for reproducibility.",
        user_default=42,
        voluntary=True,
        validator=AnyValidator([IsNoneValidator(), MinValueIntegerValidator(min_value=0)])
    )
    random_subset_size: Optional[int | float] = ConfigurableValue(
        desc="Size of the random subset of the dataset to use for calculating pseudo-perplexity. If not provided, the whole dataset will be used. use integer > 0 to define the number of samples or 0 < float < 1 to define the fraction of the dataset.",
        user_default=None,
        voluntary=True,
        validator=AnyValidator([IsNoneValidator(), MinValueIntegerValidator(min_value=1),
                                ValueInIntervalFloatValidator(min_value=0, max_value=1.0, left_inclusive=False,
                                                              right_inclusive=False)])
    )
    ignore_index: int = ConfigurableValue(
        desc="Index to ignore in the target. It is used for the loss function.",
        user_default=-100,
        voluntary=True,
        validator=IntegerValidator()
    )
    batch_size_per_device: int = ConfigurableValue(
        desc="Batch size per device for the dataloader.",
        user_default=8,
        validator=MinValueIntegerValidator(min_value=1)
    )
    pseudo_perplexity_per_sample: Optional[str] = ConfigurableValue(
        desc="Optionally you can choose path to save the pseudo-perplexity per sample in JSONL format.",
        user_default=None,
        voluntary=True,
        transform=RelativePathTransformer(allow_none=True)
    )
    attention_mask: bool = ConfigurableValue(
        desc="Whether to use attention mask in the model. If False, only input_ids will be passed to the model.",
        user_default=True,
        voluntary=True,
        validator=BoolValidator()
    )
    struct_batch: bool = ConfigurableValue(
        desc="Whether to use batching into one vector with struct indices. For evaluation set inputs in include_for_metrics.",
        user_default=False,
        voluntary=True
    )

    def __post_init__(self):
        self.input_template = None
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')
        self.accelerator = Accelerator()
        self.dataset = self.dataset_fct.create(self.split)
        self.model = self.model_fct.create()
        self._init_tokenizer()
        self.token_2_char_len = torch.tensor([len(t) for t, _ in sorted(self.tokenizer.get_vocab().items(), key=lambda x: x[1])], device=self.accelerator.device)
        self.token_2_bytes_len = torch.tensor([len(t.encode('utf-8')) for t, _ in sorted(self.tokenizer.get_vocab().items(), key=lambda x: x[1])], device=self.accelerator.device)

    def __call__(self):
        if self.random_subset_size is not None:
            number_of_samples = self.random_subset_size
            if isinstance(self.random_subset_size, int):
                if self.random_subset_size >= len(self.dataset):
                    raise ValueError("Random subset size is larger or equal to the dataset size. "
                                     "Please provide a smaller size or use the whole dataset.")
            else:
                number_of_samples = int(len(self.dataset) * self.random_subset_size)

            self.dataset = self.dataset.shuffle(seed=self.random_seed).select(range(number_of_samples))

        # we want to do this in the main process as first to avoid issues with multiprocessing

        if not self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()

        self.dataset = self.dataset.map(
            self.process_to_ppl_input,
            batched=True, batch_size=1,
            with_indices=True,
            num_proc=self.dataset_fct.num_proc if self.dataset_fct.num_proc > 0 else None,
            load_from_cache_file=True,  # to avoid reprocessing the dataset in other processes and multiple runs
            desc="Processing dataset for pseudo-perplexity calculation",
            remove_columns=self.dataset.column_names
        )

        if self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()

        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_device,
            collate_fn=self._create_collator(),
        )

        self.model, self.dataloader = self.accelerator.prepare(self.model, dataloader)
        loss_per_sample, char_loss_per_sample, bytes_loss_per_sample = self._infer_pseudo_perplexity()

        if self.accelerator.is_main_process:
            pseudo_perplexity = self.calculate_pseudo_perplexity(loss_per_sample)
            print("Average pseudo-perplexity for the dataset: ", pseudo_perplexity.mean().item())
            print("Min pseudo-perplexity for the dataset: ", pseudo_perplexity.min().item())
            print("Max pseudo-perplexity for the dataset: ", pseudo_perplexity.max().item())

            char_pseudo_perplexity = self.calculate_pseudo_perplexity(char_loss_per_sample)
            print("Average character level pseudo-perplexity for the dataset: ", char_pseudo_perplexity.mean().item())
            print("Min character level pseudo-perplexity for the dataset: ", char_pseudo_perplexity.min().item())
            print("Max character level pseudo-perplexity for the dataset: ", char_pseudo_perplexity.max().item())

            bits_per_character = self.bits_per_element(char_loss_per_sample)
            print("Average bits per character for the dataset: ", bits_per_character.mean().item())
            print("Min bits per character for the dataset: ", bits_per_character.min().item())
            print("Max bits per character for the dataset: ", bits_per_character.max().item())

            bits_per_byte = self.bits_per_element(bytes_loss_per_sample)
            print("Average bits per byte for the dataset: ", bits_per_byte.mean().item())
            print("Min bits per byte for the dataset: ", bits_per_byte.min().item())
            print("Max bits per byte for the dataset: ", bits_per_byte.max().item())

            if self.pseudo_perplexity_per_sample is not None:
                with open(self.pseudo_perplexity_per_sample, "w") as f:
                    for i, ((sample_id, (count, _)), (char_count, _), (bytes_count, _)) in enumerate(zip(loss_per_sample.items(), char_loss_per_sample.values(), bytes_loss_per_sample.values())):
                        record = {
                            "sample_index": sample_id,
                            "pseudo_perplexity": pseudo_perplexity[i].item(),
                            "char_pseudo_perplexity": char_pseudo_perplexity[i].item(),
                            "bits_per_character": bits_per_character[i].item(),
                            "bits_per_byte": bits_per_byte[i].item(),
                            "tokens": count,
                            "characters": char_count,
                            "bytes": bytes_count
                        }
                        print(
                            json.dumps(
                                record,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                            file=f
                        )

    def bits_per_element(self, loss_per_sample: dict[int, Sequence[float]]) -> torch.Tensor:
        """
        Calculates number of bits per sample.
        :param loss_per_sample:
            dictionary with sample id as a key and tuple with element count and loss sum for all elements
        :return: bits per element per each sample
        """

        counts, loss_sums = [], []
        for count, loss_sum in loss_per_sample.values():
            counts.append(count)
            loss_sums.append(loss_sum)

        counts, loss_sums = torch.tensor(counts), torch.tensor(loss_sums)

        bpe = loss_sums / (counts * math.log(2))

        return bpe

    def calculate_pseudo_perplexity(self, loss_per_sample: dict[int, Sequence[float]]) -> torch.Tensor:
        """
        Calculates pseudo perplexity for each sample
        :param loss_per_sample:
            dictionary with sample id as a key and tuple with element count and loss sum for all elements
        :return: pseudo perplexity per each sample
        """
        counts, loss_sums = [], []
        for count, loss_sum in loss_per_sample.values():
            counts.append(count)
            loss_sums.append(loss_sum)

        counts, loss_sums = torch.tensor(counts), torch.tensor(loss_sums)
        pseudo_perplexity = torch.exp(loss_sums / counts)
        return pseudo_perplexity

    def _create_collator(self):
        if self.struct_batch:
            if not hasattr(self.model.config, "levels"):
                raise ValueError("Model config does not have 'levels' attribute. "
                                 "Please set 'struct_batch' to False or use a model with 'levels' attribute.")
            lvls = self.model.config.levels
            collator = StructDataCollatorForTokenClassification(
                pursuit_minimal_context=False,
                levels=lvls,
                make_matrix_compatible=True
            )
        else:
            collator = DataCollator(self.tokenizer, attention_mask=self.attention_mask, ignore_index=self.ignore_index)

        return collator

    def _convert_struct_to_batch(self, sample_id: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, batch_index: torch.Tensor) -> tuple[list[int], list[torch.Tensor], list[torch.Tensor]]:
        """
        Converts structure batch to batch.

        :param sample_id: ids of samples in struct batch format
        :param logits: output logits in struct batch format
        :param labels: target labels in struct batch format
        :param batch_index: batch index in struct batch format
        :return:
            sample ids
            list of logits
            list of labels
        """

        b_logits, b_labels = [], []
        for b_i in range(batch_index.max().item()+1):
            m = batch_index == b_i
            b_logits.append(logits[0][m])
            b_labels.append(labels[0][m])

        return sample_id.flatten().tolist(), b_logits, b_labels

    @torch.inference_mode()
    def _infer_pseudo_perplexity(self) -> tuple[dict[int, Sequence[float]], dict[int, Sequence[float]], dict[int, Sequence[float]]]:
        """
        Calculates pseudo-perplexity for the dataset using the provided model and tokenizer.

        :return:
            Tuple of three dictionaries:
            loss_per_sample: Dictionary with sample IDs as keys and tuples of (number of elements, sum of losses) as values.
            char_loss_per_sample: Dictionary with sample IDs as keys and tuples of (number of characters, sum of character losses) as values.
            bytes_loss_per_sample: Dictionary with sample IDs as keys and tuples of (number of bytes

            It will gather results from all processes.
        """

        device = self.accelerator.device
        loss_per_sample = {}
        char_loss_per_sample = {}
        bytes_loss_per_sample = {}
        for batch in tqdm(self.dataloader, desc="Calculating pseudo-perplexity", disable=not self.accelerator.is_main_process, unit="batch"):
            model_args = {
                "input_ids": batch["input_ids"].to(device),
            }
            labels = batch["labels"].to(device)

            if self.attention_mask:
                model_args["attention_mask"] = batch["attention_mask"].to(device)

            if self.struct_batch:
                model_args["indices"] = batch["indices"].to(device)

            outputs = self.model(**model_args)

            if self.struct_batch:
                sample_ids, logits, labels = self._convert_struct_to_batch(
                    batch["sample_id"],
                    outputs.logits,
                    labels,
                    batch["indices"].batch_index
                )
            else:
                sample_ids, logits = batch["sample_id"].tolist(), outputs.logits

            for i, sample_id in enumerate(sample_ids):
                loss = self.loss(logits[i], labels[i])
                if sample_id not in loss_per_sample:
                    loss_per_sample[sample_id] = [0, 0.0]
                    char_loss_per_sample[sample_id] = [0, 0.0]
                    bytes_loss_per_sample[sample_id] = [0, 0.0]

                non_ignored_mask = labels[i].ne(self.ignore_index)

                loss = loss[non_ignored_mask]

                loss_per_sample[sample_id][0] += non_ignored_mask.sum().item()  # count only non-ignored tokens
                loss_per_sample[sample_id][1] += loss.sum().item()

                non_ignored_labels = labels[i][non_ignored_mask]

                chars_per_token = self.token_2_char_len[non_ignored_labels]
                bytes_per_token = self.token_2_bytes_len[non_ignored_labels]

                char_loss_per_sample[sample_id][0] += (non_ignored_mask * chars_per_token).sum().item()
                char_loss_per_sample[sample_id][1] += (loss * chars_per_token).sum().item()

                bytes_loss_per_sample[sample_id][0] += (non_ignored_mask * bytes_per_token).sum().item()
                bytes_loss_per_sample[sample_id][1] += (loss * bytes_per_token).sum().item()


        # gather from all processes
        loss_per_sample = self.gather_results(loss_per_sample)
        char_loss_per_sample = self.gather_results(char_loss_per_sample)
        bytes_loss_per_sample = self.gather_results(bytes_loss_per_sample)

        return loss_per_sample, char_loss_per_sample, bytes_loss_per_sample

    def gather_results(self, loss_per_sample: dict[int, Sequence[float]]) -> dict[int, Sequence[float]]:
        """
        Gathers results from all processes.

        :param loss_per_sample: Dictionary with sample IDs as keys and tuples of (number of elements, sum of losses) as values.
        :return: Dictionary with sample IDs as keys and tuples of (number of elements, sum of losses) as values.
            It will gather results from all processes.
        """
        device = self.accelerator.device

        # convert for gathering
        loss_per_sample = [[sample_id, count, loss_sum] for sample_id, (count, loss_sum) in loss_per_sample.items()]
        loss_per_sample = torch.tensor(loss_per_sample, dtype=torch.float32, device=device)

        # gather from all processes
        loss_per_sample = self.accelerator.gather_for_metrics(loss_per_sample)

        # combine results from all processes
        res = {}
        for row in loss_per_sample.tolist():
            sample_id = int(row[0])
            count = int(row[1])
            loss_sum = float(row[2])
            if sample_id not in res:
                res[sample_id] = [0, 0.0]
            res[sample_id][0] += count
            res[sample_id][1] += loss_sum
        loss_per_sample = res

        return loss_per_sample

    def _init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_fct.model_path, use_fast=True,
                                                       cache_dir=self.model_fct.cache_dir) \
            if self.tokenizer_fct is None else self.tokenizer_fct.create()

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

    def process_to_ppl_input(self, sample: dict[str, list], indices: list[int]) -> dict[str, list]:
        """
        Convert the dataset to the input format for pseudo-perplexity calculation.

        It will create multiple samples from one sample in the dataset as it will create one sample per token in the input text.

        :param sample: Sample from the dataset.
            In form of batch with one sample.
        :param indices: Index of the sample in the dataset.
        :return: Dictionary with inputs for pseudo-perplexity calculation.
        """

        input_texts = self.batch_create_input_text(sample)
        tokenized = self.tokenizer(input_texts, truncation=True, return_tensors="pt")

        # create one sample per token in the input text
        sample_len = tokenized["input_ids"].shape[1]
        input_ids = tokenized["input_ids"].repeat(sample_len, 1)

        labels = torch.where(torch.eye(sample_len, dtype=torch.bool), input_ids,
                             self.ignore_index)  # ignore index for the loss function
        input_ids = torch.where(torch.eye(sample_len, dtype=torch.bool), self.tokenizer.mask_token_id, input_ids)

        batch = {
            "input_ids": input_ids.tolist(),
            "labels": labels.tolist(),
            "sample_id": indices * sample_len
        }

        return batch


RUN = PseudoPerplexityWorkflow

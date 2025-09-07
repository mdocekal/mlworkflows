#!/usr/bin/env python

import numpy
import numpy as np
import torch
from classconfig import ConfigurableValue, ConfigurableSubclassFactory
from classconfig.validators import IntegerValidator
from transformers import DataCollatorForTokenClassification, EvalPrediction

from mlworkflows import models as mlworkflows_models
from mlworkflows.etl.collators import BatchIndexedDataCollatorForTokenClassification
from mlworkflows.utils.constants import HF_TRAINER_PAD_INDEX
from mlworkflows.utils.tokens import assemble_token_offsets, align_tokens
from mlworkflows.workflows.base import BaseTrainingWorkflow


class SequenceLabelingTrainingWorkflow(BaseTrainingWorkflow):
    model_fct: mlworkflows_models.TokenClassificationModelFactory = ConfigurableSubclassFactory(
        mlworkflows_models.TokenClassificationModelFactory,
        desc="Hugging Face model configuration.",
        name="model",
        user_default=mlworkflows_models.TokenClassificationModelFactory)
    text_field: str = ConfigurableValue(
        desc="Field in the dataset that contains the text. It will be used to create the input for the model.",
        user_default="text")
    labels_field: str = ConfigurableValue(
        desc="Field in the dataset that contains the labels. It will be used to create the target for the model.",
        user_default="labels")
    tokens_field: str = ConfigurableValue(
        desc="Field in the dataset that contains the tokens. It will be used for assembling model i/o.",
        user_default="tokens")
    start_offsets_field: str = ConfigurableValue(
        desc="Field in the dataset that contains the token start offsets. It will be used for assembling model i/o.",
        user_default="start_offsets")
    omit_label: int = ConfigurableValue(
        desc="Label that is used for special tokens that would be omited from the training and evaluation.",
        user_default=-100,
        voluntary=True,
        validator=IntegerValidator()
    )
    truncation: bool = ConfigurableValue(
        desc="Whether to truncate the input sequences to the maximum length supported by the model.",
        user_default=True
    )
    indexed_batch: bool = ConfigurableValue(
        desc="Whether to use batching into one vector with batch indices. For evaluation set inputs in include_for_metrics.",
        user_default=False,
        voluntary=True
    )

    def __post_init__(self):
        if self.compute_metrics is not None:
            self.compute_metrics.transform = self.transform_eval_pred

    def _init_components(self):
        super()._init_components()
        self.dataset = self.dataset.map(self.tokenize_and_align_labels, batched=True, load_from_cache_file=True)

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
            self.collator = BatchIndexedDataCollatorForTokenClassification(make_matrix_compatible=True)
        else:
            self.collator = DataCollatorForTokenClassification(self.tokenizer)

    def tokenize_and_align_labels(self, examples: dict) -> dict:
        """
        Tokenizes the examples and aligns the labels to the tokens.

        :param examples: Examples to tokenize.
        :return: Tokenized examples with aligned labels.
        :raises ValueError: If tokenization and label alignment fails.
        """
        if not examples:
            return {}

        # check if batch or single example
        single = False
        if isinstance(examples[self.text_field], str):
            single = True
            examples = {k: [v] for k, v in examples.items()}

        tokenized_inputs = self.tokenizer(examples[self.text_field], truncation=self.truncation,
                                          return_offsets_mapping=True)
        token_labels = []

        for batch_i in range(len(examples[self.text_field])):
            assert len(examples[self.tokens_field][batch_i]) == len(
                examples[self.labels_field][batch_i]), "Tokenization and label alignment failed."
            align_to = assemble_token_offsets(examples[self.tokens_field][batch_i],
                                                   examples[self.start_offsets_field][batch_i])
            aligning = align_tokens(tokenized_inputs["offset_mapping"][batch_i], align_to)

            labels = examples[self.labels_field][batch_i]

            token_labels.append([])
            for i, align in enumerate(aligning):
                if align == -1:
                    token_labels[-1].append(self.omit_label)
                else:
                    token_labels[-1].append(labels[align])

        tokenized_inputs[self.labels_field] = token_labels

        if single:
            tokenized_inputs = {k: v[0] for k, v in tokenized_inputs.items()}
        return tokenized_inputs

    @staticmethod
    def extract_padding_mask_from_mult_proc_gather(labels: np.ndarray, predictions: np.ndarray, elements: int) -> np.ndarray[bool]:
        """
        Extract padding mask from miltiprocesssing proc gather.
        The trouble is that the labels might also contain the same value used for padding to omit some labels from loss calculation.

        On the other hand predictions might also contain used padding value.

        :param labels: ground truth labels
        :param predictions: predicted
        :param elements: expected number of elements
        :return: padding mask
        """

        mask_from_predictions = predictions != HF_TRAINER_PAD_INDEX
        mask_from_predictions = np.all(mask_from_predictions, axis=-1)
        if mask_from_predictions.sum() == elements:
            return mask_from_predictions

        # program will probably never get here
        mask_from_labels = labels != HF_TRAINER_PAD_INDEX

        mask = mask_from_labels & mask_from_predictions
        if mask.sum() == elements:
            return mask

        # I bet this never happens, but the last try is to take last subsequent elements from right
        mask_window = numpy.zeros((mask.shape[0]), dtype=bool)
        for i in range(mask.shape[1]-1, -1, -1):
            mask_window = mask_window | mask[:, i]
            mask[:,i] = mask_window

        if mask.sum() == elements:
            return mask

        raise RuntimeError("Not able to extract padding mask")

    def transform_eval_pred(self, eval_preds: EvalPrediction):
        """
        Transforms the evaluation predictions.
            It transforms number labels to the label names.
            It filters out the omitted labels.

        :param eval_preds: Evaluation predictions.
        :return: Filtered evaluation predictions.
        """
        predictions, labels = eval_preds.predictions, eval_preds.label_ids

        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()

        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        if self.indexed_batch:
            batch_index = eval_preds.inputs["batch_index"]
            if isinstance(batch_index, torch.Tensor):
                batch_index = batch_index.cpu().numpy()

            if len(batch_index.shape) == 2 and batch_index.shape[0] == 1:
                # the batch_index is not gathered by trainer
                batch_index = self.trainer.gather_function((batch_index,))

                # concatenate
                concatenated_batch_index = np.empty((sum(s.shape[-1] for s in batch_index), ), batch_index[0].dtype)
                batch_sample_offset = 0
                batch_offset = 0
                for s in batch_index:
                    concatenated_batch_index[batch_offset:batch_offset+s.shape[1]] = s + batch_sample_offset
                    batch_sample_offset += np.max(s) + 1
                    batch_offset += s.shape[1]
                batch_index = concatenated_batch_index

                # remove padding which was added by Hugging Face trainer during gathering
                padding_mask = self.extract_padding_mask_from_mult_proc_gather(labels, predictions,
                                                                               batch_index.shape[0])
                predictions = predictions[padding_mask]
                labels = labels[padding_mask]

            predictions = np.argmax(predictions, axis=1)  # the indexed_batch is for compatibility issues convert to PxN matrix, so axis 2 is also ok

            max_batch_index = numpy.max(batch_index, axis=0)
            per_sample_predictions, per_sample_labels = [], []
            for i in range(max_batch_index + 1):
                selection_mask = batch_index == i

                per_sample_predictions.append(predictions[selection_mask])
                per_sample_labels.append(labels[selection_mask])

            predictions, labels = per_sample_predictions, per_sample_labels
        else:
            predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [self.model_fct.labels[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        true_labels = [
            [self.model_fct.labels[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        eval_preds.predictions = true_predictions
        eval_preds.label_ids = true_labels


RUN = SequenceLabelingTrainingWorkflow

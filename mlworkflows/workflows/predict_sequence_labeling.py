#!/usr/bin/env python
import json
from pathlib import Path
from typing import Sequence, Dict, Any, Optional, Generator

import numpy
import numpy as np
import torch
from classconfig import ConfigurableValue, ConfigurableSubclassFactory
from classconfig.validators import IntegerValidator
from evaluate import Metric

from transformers import DataCollatorForTokenClassification, EvalPrediction, TrainingArguments
from transformers.trainer_utils import PredictionOutput

from mlworkflows import models as mlworkflows_models
from mlworkflows.etl.collators import BatchIndexedDataCollatorForTokenClassification
from mlworkflows.utils.constants import HF_TRAINER_PAD_INDEX
from mlworkflows.utils.tokens import assemble_token_offsets, align_tokens
from mlworkflows.workflows.base import BaseTrainingWorkflow, BasePredictWorkflow


class FakeMetricForResults:
    """
    In case of indexed batch we use dirty hack to get the results from the evaluation. We will use this fake metric to store the results as
    only the metric is able to obtain all inputs.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.results = []
        self.transform = None

    def __call__(self, eval_preds: EvalPrediction, compute_result: bool = True) -> Optional[dict]:
        """
        Computes evaluation metrics.

        :param eval_preds: Evaluation predictions.
        :param compute_result: Whether to compute the results.
        :return: Metrics.
        """
        if self.transform is not None:
            self.transform(eval_preds)

        self.results.append(eval_preds)

        if compute_result:
            return {"results": len(self.results)}
        return None


class SequenceLabelingPredictWorkflow(BasePredictWorkflow):

    model_fct: mlworkflows_models.TokenClassificationModelFactory = ConfigurableSubclassFactory(
        mlworkflows_models.TokenClassificationModelFactory,
        desc="Hugging Face model configuration.",
        name="model",
        user_default=mlworkflows_models.TokenClassificationModelFactory)
    text_field: str = ConfigurableValue(
        desc="Field in the dataset that contains the text. It will be used to create the input for the model.",
        user_default="text")
    tokens_field: str = ConfigurableValue(
        desc="Field in the dataset that contains the tokens. It will be used for assembling model i/o.",
        user_default="tokens")
    start_offsets_field: str = ConfigurableValue(
        desc="Field in the dataset that contains the token start offsets. It will be used for assembling model i/o.",
        user_default="start_offsets")
    id_field: Optional[str] = ConfigurableValue(
        desc="Field in the dataset that contains the id. It will be used for assembling model i/o if not None.",
        user_default=None,
        voluntary=True
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
        self.compute_metrics = FakeMetricForResults()
        if self.compute_metrics is not None:
            self.compute_metrics.transform = self.transform_eval_pred

    def _init_dataset(self):
        self.dataset = self.dataset_fct.create()[self.use_split]

    def _init_trainer_args(self) -> TrainingArguments:
        args = super()._init_trainer_args()

        if self.indexed_batch:
            args["compute_metrics"] = self.compute_metrics
            args["include_for_metrics"] = ["inputs"]
            args["batch_eval_metrics"] = True

        return args

    def _init_components(self):
        super()._init_components()
        self.dataset = self.dataset.map(self.tokenize, batched=True, load_from_cache_file=True)

    def _init_tokenizer(self):
        super()._init_tokenizer()
        if not self.tokenizer.is_fast:
            raise ValueError("Tokenizer is not fast. Please use model with fast tokenizer.")

    def _init_model(self):
        self.model = self.model_fct.create()

    def _init_collator(self):
        if self.indexed_batch:
            self.collator = BatchIndexedDataCollatorForTokenClassification(make_matrix_compatible=True)
        else:
            self.collator = DataCollatorForTokenClassification(self.tokenizer)

    def tokenize(self, examples: dict) -> dict:
        """
        Tokenizes the examples.

        :param examples: Examples to tokenize.
        :return: Tokenized examples
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
        
        tokenized_inputs["labels"] = [[0] * len(t) for t in tokenized_inputs["input_ids"]]  # dummy labels

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

            # remove padding
            mask = predictions != HF_TRAINER_PAD_INDEX
            mask = np.all(mask, axis=-1)

            predictions = np.argmax(predictions, axis=2)
            per_sample_predictions, per_sample_labels = [], []

            for i in range(mask.shape[0]):
                per_sample_predictions.append(predictions[i][mask[i]])
                per_sample_labels.append(labels[i][mask[i]])

            predictions, labels = per_sample_predictions, per_sample_labels

        eval_preds.predictions = predictions
        eval_preds.label_ids = labels

    def transform_token_predictions(self, predictions: np.ndarray, sample_index: int) -> list[int]:
        """
        Aligns token level predictions with labels in the dataset.

        :param predictions: Predictions.
        :param sample_index: Index of the sample.
        :return: Dataset aligned predictions.
        """

        dataset_sample = self.dataset[sample_index]

        original_offsets = assemble_token_offsets(
            dataset_sample[self.tokens_field],
            dataset_sample[self.start_offsets_field]
        )

        alignment = align_tokens(
            dataset_sample["offset_mapping"],
            original_offsets
        )

        aligned_predictions = [0] * len(original_offsets)

        for a, p in zip(alignment, predictions):
            if a != -1:
                aligned_predictions[a] = p.item()

        return aligned_predictions

    def __call__(self):
        trainer = self._init_trainer()
        res = trainer.predict(self.dataset)
        if self.indexed_batch:
            self.write_predictions_indexed_batch(self.compute_metrics.results)
        else:
            self.write_predictions(res)

    def write_predictions_indexed_batch(self, res: Sequence[EvalPrediction]):
        if self.predictions is not None:
            test_dataset = self.dataset
            i = 0
            with open(Path(self.predictions) / "results.jsonl", "w") as f:
                for r in res:
                    for pred in r.predictions:
                        res = {}
                        if self.id_field is not None:
                            res["id"] = test_dataset[i][self.id_field]
                        res["labels"] = self.transform_token_predictions(pred, i)
                        print(json.dumps(res), file=f)
                        i += 1

    def transform_predictions(self, predictions: PredictionOutput) -> Generator[dict[str, Any], None, None]:
        if self.indexed_batch:
            raise RuntimeError("The program should never get here.")

        transformed = EvalPrediction(
            predictions=predictions.predictions,
            label_ids=predictions.label_ids
        )
        self.transform_eval_pred(transformed)

        test_dataset = self.dataset
        for i, pred in enumerate(transformed.predictions):
            res = {}
            if self.id_field is not None:
                res["id"] = test_dataset[i][self.id_field]

            res["labels"] = self.transform_token_predictions(pred, i)
            yield res


RUN = SequenceLabelingPredictWorkflow

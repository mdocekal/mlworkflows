# -*- coding: UTF-8 -*-
"""
Created on 20.01.25

:author:     Martin Dočekal
"""
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
import torch
from transformers import EvalPrediction

from mlworkflows.workflows.train_sequence_labeling import SequenceLabelingTrainingWorkflow


class TestTokenizeAndAlignLabels(TestCase):
    def setUp(self):
        self.tokenizer = MagicMock()
        tokenizer_fct = MagicMock()
        tokenizer_fct.create.return_value = self.tokenizer

        self.workflow = SequenceLabelingTrainingWorkflow(
            dataset_fct=MagicMock(),
            model_fct=MagicMock(),
            tokenizer_fct=tokenizer_fct,
            training=MagicMock(),
            compute_metrics=None
        )
        self.workflow._init_components()

    def test_empty(self):
        self.assertDictEqual(
            {},
            self.workflow.tokenize_and_align_labels({})
        )

    def test_single(self):
        self.tokenizer.return_value = {
            "input_ids": [[1, 2, 3, 4, 5]],
            "offset_mapping": [[(0, 2), (3, 7), (8, 10), (11, 17), (18, 22)]]
        }

        self.assertDictEqual(
            {
                "input_ids": [1, 2, 3, 4, 5],
                "labels": [0, 0, 0, 1, 2],
                "offset_mapping": [(0, 2), (3, 7), (8, 10), (11, 17), (18, 22)]
            },
            self.workflow.tokenize_and_align_labels(
                {
                    "text": "My name is Dorian Gray",
                    "labels": [0, 0, 0, 1, 2],
                    "tokens": ["My", "name", "is", "Dorian", "Gray"],
                    "start_offsets": [0, 3, 8, 11, 18]
                },
            )
        )

    def test_mismatching_tokens(self):
        self.tokenizer.return_value = {
            "input_ids": [[1, 2, 3, 4, 5]],
            "offset_mapping": [[(0, 2), (3, 7), (8, 10), (11, 20), (21, 22)]]
        }
        self.assertDictEqual(
            {
                "input_ids": [1, 2, 3, 4, 5],
                "labels": [0, 0, 0, 1, 2],
                "offset_mapping": [(0, 2), (3, 7), (8, 10), (11, 20), (21, 22)]
            },
            self.workflow.tokenize_and_align_labels(
                {
                    "text": "My name is Dorian Gray",
                    "labels": [0, 0, 0, 1, 2],
                    "tokens": ["My", "name", "is", "Dorian", "Gray"],
                    "start_offsets": [0, 3, 8, 11, 18]
                },
            )
        )

    def test_batch(self):
        self.tokenizer.return_value = {
            "input_ids": [[1, 2, 3, 4, 5], [6, 7, 8, 9]],
            "offset_mapping": [[(0, 2), (3, 7), (8, 10), (11, 17), (18, 22)], [(0, 5), (5, 6), (7, 12), (12, 13)]]
        }

        self.assertDictEqual(
            {
                "input_ids": [[1, 2, 3, 4, 5], [6, 7, 8, 9]],
                "labels": [[0, 0, 0, 1, 2], [0, 1, 2, 0]],
                "offset_mapping": [[(0, 2), (3, 7), (8, 10), (11, 17), (18, 22)], [(0, 5), (5, 6), (7, 12), (12, 13)]]
            },
            self.workflow.tokenize_and_align_labels(
                {
                    "text": ["My name is Dorian Gray", "Hello, world!"],
                    "labels": [[0, 0, 0, 1, 2], [0, 1, 2, 0]],
                    "tokens": [["My", "name", "is", "Dorian", "Gray"], ["Hello", ",", "world", "!"]],
                    "start_offsets": [[0, 3, 8, 11, 18], [0, 5, 7, 12]]
                },
            )
        )


class TestExtractPaddingMaskFromMultProcGather(TestCase):

    def test_extract_from_predictions(self):
        labels = np.array(
            [
                [1, 2, 3],
                [1, 4, -100]
            ]
        )
        predictions = np.array(
            [
                [
                    [0.1, 0.2, 0.7],
                    [0.1, 0.2, 0.7],
                    [0.1, 0.2, 0.7]
                ],
                [
                    [0.1, 0.2, 0.7],
                    [0.1, 0.2, 0.7],
                    [-100, -100, -100]
                ]
            ]
        )

        mask = SequenceLabelingTrainingWorkflow.extract_padding_mask_from_mult_proc_gather(labels, predictions, 5)
        self.assertSequenceEqual(
            [
                [True, True, True],
                [True, True, False]
            ],
            mask.tolist()
        )

    def test_extract_from_labels(self):
        labels = np.array(
            [
                [1, 2, 3],
                [1, 4, -100]
            ]
        )
        predictions = np.array(
            [
                [
                    [0.1, 0.2, 0.7],
                    [0.1, 0.2, 0.7],
                    [0.1, 0.2, 0.7]
                ],
                [
                    [-100, -100, -100],
                    [0.1, 0.2, 0.7],
                    [-100, -100, -100]
                ]
            ]
        )

        mask = SequenceLabelingTrainingWorkflow.extract_padding_mask_from_mult_proc_gather(labels, predictions, 5)
        self.assertSequenceEqual(
            [
                [True, True, True],
                [True, True, False]
            ],
            mask.tolist()
        )

    def test_extract_from_last_subsequent_elements(self):
        labels = np.array(
            [
                [1, 2, 3],
                [-100, 4, -100]
            ]
        )
        predictions = np.array(
            [
                [
                    [0.1, 0.2, 0.7],
                    [0.1, 0.2, 0.7],
                    [0.1, 0.2, 0.7]
                ],
                [
                    [-100, -100, -100],
                    [0.1, 0.2, 0.7],
                    [-100, -100, -100]
                ]
            ]
        )

        mask = SequenceLabelingTrainingWorkflow.extract_padding_mask_from_mult_proc_gather(labels, predictions, 5)
        self.assertSequenceEqual(
            [
                [True, True, True],
                [True, True, False]
            ],
            mask.tolist()
        )


class TestTransformEvalPred(TestCase):
    def setUp(self):
        self.tokenizer = MagicMock()
        tokenizer_fct = MagicMock()
        tokenizer_fct.create.return_value = self.tokenizer

        self.workflow = SequenceLabelingTrainingWorkflow(
            dataset_fct=MagicMock(),
            model_fct=MagicMock(),
            tokenizer_fct=tokenizer_fct,
            training=MagicMock(),
            compute_metrics=None,
        )
        self.workflow._init_components()
        self.workflow.model_fct.labels.__getitem__.side_effect = lambda x: ["B", "I", "O"][x]

    def test_transform_eval_pred(self):
        e = EvalPrediction(
            predictions=torch.tensor([
                [
                    [0.9, 0.2, 0.7],
                    [0.1, 0.9, 0.7],
                    [0.1, 0.2, 0.7]
                ],
                [
                    [0.1, 0.2, 0.7],
                    [0.1, 0.2, 0.7],
                    [-100, -100, -100]
                ]
            ]),
            label_ids=torch.tensor([
                [0, 1, 2],
                [0, 1, -100]
            ])
        )
        self.workflow.transform_eval_pred(e)

        self.assertEqual(
            [
                [
                    "B", "I", "O"
                ],
                [
                    "O", "O"
                ]
            ],
            e.predictions
        )
        self.assertEqual(
            [
                [
                    "B", "I", "O"
                ],
                [
                    "B", "I"
                ]
            ],
            e.label_ids
        )

    def test_transform_eval_pred_indexed_batch(self):
        self.workflow.indexed_batch = True
        e = EvalPrediction(
            predictions=torch.tensor([
                [0.9, 0.2, 0.7],
                [0.1, 0.9, 0.7],
                [0.1, 0.2, 0.7],
                [0.1, 0.2, 0.7],
                [0.1, 0.2, 0.7]
            ]),
            label_ids=torch.tensor([0, 1, 2, 0, 1]),
            inputs={
                "batch_index": torch.tensor([0, 0, 0, 1, 1])
            }
        )
        self.workflow.transform_eval_pred(e)

        self.assertEqual(
            [
                [
                    "B", "I", "O"
                ],
                [
                    "O", "O"
                ]
            ],
            e.predictions
        )
        self.assertEqual(
            [
                [
                    "B", "I", "O"
                ],
                [
                    "B", "I"
                ]
            ],
            e.label_ids
        )

    def test_transform_eval_pred_indexed_batch_multi_proc(self):
        self.workflow.indexed_batch = True
        self.workflow.trainer = MagicMock()
        self.workflow.trainer.gather_function.return_value = (np.array([[0, 0, 0]]), np.array([[0, 0]]))
        e = EvalPrediction(
            predictions=torch.tensor([
                [
                    [0.9, 0.2, 0.7],
                    [0.1, 0.9, 0.7],
                    [0.1, 0.2, 0.7]
                ],
                [
                    [0.1, 0.2, 0.7],
                    [0.1, 0.2, 0.7],
                    [-100, -100, -100]
                ]
            ]),
            label_ids=torch.tensor([
                [0, 1, 2],
                [0, 1, -100]
            ]),
            inputs={
                "batch_index": torch.tensor([[0, 0, 0]])
            }
        )
        self.workflow.transform_eval_pred(e)

        self.assertEqual(
            [
                [
                    "B", "I", "O"
                ],
                [
                    "O", "O"
                ]
            ],
            e.predictions
        )
        self.assertEqual(
            [
                [
                    "B", "I", "O"
                ],
                [
                    "B", "I"
                ]
            ],
            e.label_ids
        )






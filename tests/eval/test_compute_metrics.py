# -*- coding: UTF-8 -*-
"""
Created on 20.01.25

:author:     Martin Dočekal
"""
from unittest import TestCase

import numpy as np
from transformers import EvalPrediction

from mlworkflows.eval.compute_metrics import ComputeMetrics


class TestComputeMetrics(TestCase):
    def test_accuracy(self):
        """
        Test of the accuracy method.
        """
        m = ComputeMetrics("accuracy")
        res = m(EvalPrediction(predictions=np.array([0, 1, 1, 0]), label_ids=np.array([1, 1, 1, 1])))
        self.assertDictEqual({"accuracy": 0.5}, res)

        res = m(EvalPrediction(predictions=np.array([0, 1, 1, 0]), label_ids=np.array([1, 1, 1, 0])))
        self.assertDictEqual({"accuracy": 0.75}, res)

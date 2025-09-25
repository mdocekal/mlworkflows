import unittest
from unittest.mock import MagicMock

from mlworkflows.etl.collators import MaskedLanguageModelingFlatteningDataCollator


class TestMaskedLanguageModelingFlatteningDataCollator(unittest.TestCase):
    def setUp(self):
        self.collator = MaskedLanguageModelingFlatteningDataCollator(
            tokenizer=MagicMock(),
            selection_prob=0.0,
            span_selection_prob=0.0,
            masking_prob=0.0,
            random_prob=0.0,
            span_sep_token=None,
            ignore=-100
        )

    def test_flattening(self):
        features = [
            {'input_ids': [1, 2, 3]},
            {'input_ids': [4, 5]}
        ]
        flattened = self.collator(features)
        self.assertSequenceEqual(flattened['input_ids'].tolist(), [[1, 2, 3, 4, 5]])
        self.assertSequenceEqual(flattened['labels'].tolist(), [[-100, -100, -100, -100, -100]])
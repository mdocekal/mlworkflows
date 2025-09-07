import unittest
from unittest import mock

import torch

from mlworkflows.etl.transform import masked_language_modeling_masks, mlm_transform


class TestMaskedLanguageModelingMasks(unittest.TestCase):
    trials = 1000

    def test_masked_language_modeling_masks(self):
        input_ids = torch.range(0, 100)

        selected_sum, masking_mask_sum, random_mask_sum = 0, 0, 0
        for _ in range(self.trials):
            selected, masking_mask, random_mask = masked_language_modeling_masks(
                input_ids,
                selection_prob=0.1,
                span_selection_prob=0.0,
                masking_prob=0.8,
                random_prob=0.1,
                span_sep_token=None
            )
            self.assertEqual(selected.shape, input_ids.shape)
            self.assertEqual(masking_mask.shape, input_ids.shape)
            self.assertEqual(random_mask.shape, input_ids.shape)
            self.assertEqual(selected.dtype, torch.bool)
            self.assertEqual(masking_mask.dtype, torch.bool)
            self.assertEqual(random_mask.dtype, torch.bool)

            self.assertFalse((masking_mask & random_mask).any())
            self.assertFalse((~selected & masking_mask).any())
            self.assertFalse((~selected & random_mask).any())

            sel_sum = selected.sum().item()
            mask_sum = masking_mask.sum().item()
            ran_sum = random_mask.sum().item()

            self.assertGreaterEqual(sel_sum, mask_sum)
            self.assertGreaterEqual(sel_sum, ran_sum)

            selected_sum += sel_sum
            masking_mask_sum += mask_sum
            random_mask_sum += ran_sum

        # this is P(selected)
        self.assertAlmostEqual(selected_sum / input_ids.shape[0] / self.trials, 0.1, delta=0.05,
                               msg="Statistical test failed. Even though the failure for correct code is very improbable it is possible.")

        # this is P(masked) * P(selected) = 0.8 * 0.1 = 0.08
        self.assertAlmostEqual(masking_mask_sum / input_ids.shape[0] / self.trials, 0.08, delta=0.005,
                                 msg="Statistical test failed. Even though the failure for correct code is very improbable it is possible.")

        # this is P(random) * P(selected) * P(not masked) = 0.1 * 0.1 * 0.2 = 0.002
        self.assertAlmostEqual(random_mask_sum / input_ids.shape[0] / self.trials, 0.002, delta=0.0005,
                                 msg="Statistical test failed. Even though the failure for correct code is very improbable it is possible.")

    def test_masked_language_modeling_masks_all_spans(self):
        input_ids = torch.tensor([0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 0, 16, 17, 18, 19, 20])

        selected, masking_mask, random_mask = masked_language_modeling_masks(
            input_ids,
            selection_prob=0.0,
            span_selection_prob=1.0,
            masking_prob=0.8,
            random_prob=0.1,
            span_sep_token=0
        )
        self.assertSequenceEqual(selected.tolist(), (input_ids != 0).tolist())

    def test_masked_language_modeling_masks_spans(self):
        input_ids = torch.tensor([0, 1, 1, 1, 0, 2, 2, 2, 2, 0, 3, 3, 0, 4, 4, 4, 4, 4])
        proper_counts = {
            1: 3,
            2: 4,
            3: 2,
            4: 5
        }
        starts_mask = input_ids == 0

        for _ in range(self.trials):
            selected, masking_mask, random_mask = masked_language_modeling_masks(
                input_ids,
                selection_prob=0.0,
                span_selection_prob=0.1,
                masking_prob=0.8,
                random_prob=0.1,
                span_sep_token=0
            )
            if selected.sum() == 0:
                continue

            self.assertFalse((selected & starts_mask).any())

            unique, counts = torch.unique(input_ids[selected], return_counts=True)

            for u, c in zip(unique, counts):
                self.assertEqual(c.item(), proper_counts[u.item()])


class TestMLMTransform(unittest.TestCase):
    """
    Tests for the masked language modeling transform.
    """

    @mock.patch('mlworkflows.etl.transform.torch.randint')
    @mock.patch('mlworkflows.etl.transform.masked_language_modeling_masks')
    def test_mlm_transform(self, mock_masks, mock_randint):
        """
        Test the masked language modeling transform.
        This is a placeholder for the actual test implementation.
        """
        features = {
            "input_ids": torch.tensor([1, 2, 3, 4, 5])
        }
        mock_masks.return_value = (
            torch.tensor([True, False, True, False, True]),   # selected
            torch.tensor([True, False, True, False, False]),  # masking
            torch.tensor([False, False, False, False, True])  # random
        )
        mock_randint.return_value = torch.tensor([42])
        mlm_transform(
            features=features,
            selection_prob=0.1,
            masking_prob=0.8,
            random_prob=0.1,
            span_selection_prob=0.0,
            vocab_size=1000,
            mask_token_id=100,
            span_sep_token=None,
            ignore=-100,
            unselectable_token_id=None
        )

        mock_masks.assert_called_with(
            features["input_ids"],
            selection_prob=0.1,
            span_selection_prob=0.0,
            masking_prob=0.8,
            random_prob=0.1,
            span_sep_token=None,
            unselectable_token_id=None
        )

        self.assertTrue("labels" in features)
        self.assertSequenceEqual(
            [1, -100, 3, -100, 5],
            features["labels"].tolist()
        )
        self.assertSequenceEqual(
            [100, 2, 100, 4, 42],
            features["input_ids"].tolist()
        )



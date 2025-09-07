from pathlib import Path
from unittest import TestCase, mock

from mlworkflows.etl.dataset import DatasetFactory, DatasetSplit

SCRIPT_PATH = Path(__file__).resolve().parents[0]
FIXTURES_PATH = SCRIPT_PATH / "fixtures"
DATASET_PATH = FIXTURES_PATH / "dataset.jsonl"


class TestDatasetFactory(TestCase):
    def test_local(self):
        dataset_fct = DatasetFactory(path="json", data_files=str(DATASET_PATH))

        dataset = dataset_fct.create()

        self.assertEqual(10, len(dataset[dataset_fct.train_split]))
        self.assertDictEqual(
            {"id": 7, "text": "I'm satisfied with my purchase.", "sentiment": "positive"},
            dataset[dataset_fct.train_split][7]
        )

    @mock.patch("mlworkflows.etl.dataset.load_dataset")
    def test_remote(self, mock_load_dataset):
        dataset_fct = DatasetFactory(path="amazon_polarity", name="negative", data_files=None)

        self.assertEqual(0, mock_load_dataset.call_count)
        _ = dataset_fct.create()
        mock_load_dataset.assert_called_once_with("amazon_polarity", "negative", data_dir=None, data_files=None, cache_dir=None,
                                                  split=None, streaming=False, num_proc=None)




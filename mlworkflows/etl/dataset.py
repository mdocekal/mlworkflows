from enum import Enum
from typing import Union, Optional

from classconfig import ConfigurableMixin, ConfigurableValue, RelativePathTransformer
from classconfig.transforms import CPUWorkersTransformer
from classconfig.validators import StringValidator, AnyValidator, IsNoneValidator
from datasets import load_dataset, DatasetDict, Dataset, IterableDatasetDict, IterableDataset, load_from_disk


class DatasetSplit(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class DatasetFactory(ConfigurableMixin):
    path: str = ConfigurableValue(desc="Path or HF dataset name of the dataset.",
                                  transform=RelativePathTransformer(force_relative_prefix=True))
    name: Optional[str] = ConfigurableValue(desc="Name/configuration of the dataset.", user_default=None,
                                            validator=AnyValidator([IsNoneValidator(), StringValidator()]))
    data_dir: Optional[str] = ConfigurableValue(desc="Defining the `data_dir` of the dataset configuration. If specified for the generic builders (csv, text etc.) or the Hub datasets and `data_files` is `None`, the behavior is equal to passing `os.path.join(data_dir, **)` as `data_files` to reference all the files in a directory.",
                                                user_default=None,
                                                validator=AnyValidator([IsNoneValidator(), StringValidator()]))
    data_files: Optional[str] = ConfigurableValue(
        desc="Path to source data file. In contrast to Hugging Face loader only one file is supported.",
        user_default=None,
        transform=RelativePathTransformer(force_relative_prefix=True, allow_none=True),
    )

    cache_dir: Optional[str] = ConfigurableValue(desc="Cache directory for the dataset.", user_default=None,
                                       transform=RelativePathTransformer(force_relative_prefix=True, allow_none=True))
    train_split: str = ConfigurableValue(desc="Name of the training split.", user_default="train",
                                         validator=StringValidator())
    validation_split: str = ConfigurableValue(desc="Name of the validation split.", user_default="validation",
                                              validator=StringValidator())
    test_split: str = ConfigurableValue(desc="Name of the test split.", user_default="test",
                                        validator=StringValidator())
    streaming: bool = ConfigurableValue(
        desc="Whether to load the dataset in streaming mode. If True, the dataset will be loaded as an IterableDataset.",
        user_default=False,
        voluntary=True,
    )
    num_proc: int = ConfigurableValue(
        desc="Number of processes to use for loading the dataset. If 0, the multiprocessing will not be used.",
        user_default=0,
        voluntary=True,
        transform=CPUWorkersTransformer()
    )
    load_saved: bool = ConfigurableValue(
        desc="If True, the dataset will be loaded with `load_from_disk`. Useful for loading datasets that were previously saved with `save_to_disk`.",
        user_default=False,
        voluntary=True,
    )
    preprocessed: bool = ConfigurableValue(
        desc="If True, the dataset is already preprocessed and does not need to be processed again.",
        user_default=False,
        voluntary=True,
    )

    def __post_init__(self):
        self.singleton_dataset = None

    def create(self, split: Optional[str] = None) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
        num_proc = self.num_proc if self.num_proc > 0 else None

        if self.load_saved:
            d = load_from_disk(self.path)
            return d[split] if split else d

        if self.path.endswith(".json") or self.path.endswith(".jsonl"):
            return load_dataset("json", data_files=self.path, num_proc=num_proc)
        elif self.path.endswith(".csv"):
            return load_dataset("csv", data_files=self.path, num_proc=num_proc)
        elif self.path.endswith(".txt"):
            return load_dataset("text", data_files=self.path, num_proc=num_proc)
        else:
            return load_dataset(self.path,
                                self.name,
                                data_dir=self.data_dir,
                                data_files=self.data_files,
                                cache_dir=self.cache_dir,
                                split=split,
                                streaming=self.streaming,
                                num_proc=num_proc
                                )

    def get_split(self, split: DatasetSplit, force_reload=False) -> Union[Dataset, IterableDataset]:
        """
        Gets the dataset split.

        :param split: Split to get.
        :param force_reload: Whether to force reload the dataset or use the singleton instance.
        :return: HF dataset split.
        """

        if force_reload or self.singleton_dataset is None:
            self.singleton_dataset = self.create()

        if split == DatasetSplit.TRAIN:
            return self.singleton_dataset[self.train_split]
        elif split == DatasetSplit.VALIDATION:
            return self.singleton_dataset[self.validation_split]
        elif split == DatasetSplit.TEST:
            return self.singleton_dataset[self.test_split]
        else:
            raise ValueError(f"Unknown dataset split: {split}")

    def get_train_split(self, force_reload=False) -> Union[Dataset, IterableDataset]:
        return self.get_split(DatasetSplit.TRAIN, force_reload)

    def get_validation_split(self, force_reload=False) -> Union[Dataset, IterableDataset]:
        return self.get_split(DatasetSplit.VALIDATION, force_reload)

    def get_test_split(self, force_reload=False) -> Union[Dataset, IterableDataset]:
        return self.get_split(DatasetSplit.TEST, force_reload)


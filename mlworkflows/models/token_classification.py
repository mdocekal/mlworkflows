from typing import Optional, Union

from classconfig import ConfigurableValue
from classconfig.validators import AnyValidator, IsNoneValidator, ListOfTypesValidator
from datasets import Dataset, IterableDataset
from transformers import PreTrainedModel, AutoModelForTokenClassification

from mlworkflows.models.hf_base import HFModelFactory


class TokenClassificationModelFactory(HFModelFactory):
    """
    Factory for transformers token classification model.
    """
    LOADER = AutoModelForTokenClassification
    labels: Optional[list[str]] = ConfigurableValue(
        desc="Classification labels, the position is specifying label id. Leave empty for automatic detection of labels from dataset or using labels from model configuration.",
        validator=AnyValidator([IsNoneValidator(), ListOfTypesValidator(str, allow_empty=True)])
    )

    def get_labels_from_dataset(self, dataset: Dataset, labels_feature: str) -> list[str]:
        """
        Extracts labels from the dataset.

        :param dataset: dataset to extract labels from.
        :param labels_feature: name of the feature in the dataset that contains the labels.
        :return: list of labels.
        """
        return dataset.features[labels_feature].feature.names

    def get_model_args(self, dataset: Optional[Union[Dataset, IterableDataset]] = None,
                       labels_feature: Optional[str] = None, **kwargs) -> dict:
        """
        Returns arguments for model creation.

        :param dataset: dataset that can be used to identify labels.
        :param labels_feature: name of the feature in the dataset that contains the labels.
        :param kwargs: Additional arguments that will be used to update the model arguments obtained from the factory configuration.
        :return: model arguments.
        """

        labels = self.labels
        if labels is None or len(labels) == 0:
            if labels_feature is not None:
                if dataset is None:
                    raise ValueError("Dataset is required to identify labels.")
                labels = self.get_labels_from_dataset(dataset, labels_feature)
                self.labels = labels

        if labels:  # if labels are specified, update the model arguments
            kwargs["num_labels"] = len(labels)
            kwargs["id2label"] = {i: label for i, label in enumerate(labels)}
            kwargs["label2id"] = {label: i for i, label in enumerate(labels)}
        additional_args_for_model = super().get_model_args(**kwargs)
        return additional_args_for_model

    def create(self, dataset: Optional[Union[Dataset, IterableDataset]] = None, labels_feature: Optional[str] = None,
               **kwargs) -> PreTrainedModel:
        """
        Creates the model.

        :param dataset: dataset that can be used to identify labels.
            It will obtain it from features[labels_feature].feature.names.
        :param labels_feature: name of the feature in the dataset that contains the labels.
        :return: initialized model
        :raises ValueError: if labels are not specified and dataset is not provided.
        """

        return super().create(dataset=dataset, labels_feature=labels_feature, **kwargs)

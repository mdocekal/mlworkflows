from transformers import AutoModelForSequenceClassification

from mlworkflows.models import TokenClassificationModelFactory


class SequenceClassificationModelFactory(TokenClassificationModelFactory):
    """
    Factory for transformers token classification model.
    """
    LOADER = AutoModelForSequenceClassification

    def get_labels_from_dataset(self, dataset, labels_feature):
        return dataset.features[labels_feature].names

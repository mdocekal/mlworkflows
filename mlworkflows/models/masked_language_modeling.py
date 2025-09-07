from transformers import AutoModelForMaskedLM

from mlworkflows.models.hf_base import HFModelFactory


class MaskedLanguageModelFactory(HFModelFactory):
    """
    Factory for transformers masked language model.
    """
    LOADER = AutoModelForMaskedLM

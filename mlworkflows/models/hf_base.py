from abc import ABC
from typing import Optional

from classconfig import ConfigurableMixin, ConfigurableValue, RelativePathTransformer, ConfigurableFactory
from transformers import PreTrainedModel, AutoModel, PretrainedConfig, AutoConfig

from mlworkflows.models.quantization import ConfigurableBitsAndBytesFactory


class HFModelFactory(ConfigurableMixin, ABC):
    """
    Base factory for creating HF models.
    """
    LOADER = AutoModel
    CONFIG_LOADER = AutoConfig

    model_path: str = ConfigurableValue("Name or path to the model.",
                                        transform=RelativePathTransformer(force_relative_prefix=True))

    attn_implementation: Optional[str] = ConfigurableValue(
        'The attention implementation to use in the model (if relevant). Can be any of "eager" (manual implementation of the attention), "sdpa" (using F.scaled_dot_product_attention), or "flash_attention_2" (using Dao-AILab/flash-attention). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual "eager" implementation.',
        user_default=None,
        voluntary=True)

    cache_dir: Optional[str] = ConfigurableValue("Path to Hugging Face cache directory.", user_default=None,
                                                 voluntary=True)

    quantization: Optional[ConfigurableBitsAndBytesFactory] = ConfigurableFactory(ConfigurableBitsAndBytesFactory,
                                                                                  "Configuration for bits and bytes quantization.",
                                                                                  voluntary=True)
    torch_dtype: Optional[str] = ConfigurableValue(
        "Override the default torch.dtype and load the model under a specific dtype",
        user_default=None,
        voluntary=True)
    trust_remote_code: Optional[bool] = ConfigurableValue(
        "Whether to trust remote code.",
        user_default=False,
        voluntary=True
    )
    config: Optional[PretrainedConfig] = ConfigurableValue("Configuration for the model.", user_default=None,
                                                              voluntary=True)
    random_initialization: bool = ConfigurableValue(
        "If True, the model will be initialized randomly instead of loading from the pretrained model.",
        user_default=False,
        voluntary=True
    )

    def __init__(self, **kwargs):
        """
        Initializes the factory.

        :param kwargs: Factory configuration.
        """
        ConfigurableMixin.__init__(self, **kwargs)
        self.was_quantized = False

    def get_model_args(self, **kwargs) -> dict:
        """
        Returns arguments for model creation.

        :param kwargs: Additional arguments that will be used to update the model arguments obtained from the factory configuration.
        :return: model arguments.
        """
        additional_args_for_model = {}
        if isinstance(self.quantization, ConfigurableBitsAndBytesFactory):
            q = self.quantization.create()
            self.was_quantized = q.load_in_4bit or q.load_in_8bit
            if self.was_quantized:
                additional_args_for_model["quantization_config"] = q

        if self.torch_dtype is not None:
            additional_args_for_model["torch_dtype"] = self.torch_dtype

        if self.attn_implementation is not None:
            additional_args_for_model["attn_implementation"] = self.attn_implementation

        if self.cache_dir is not None:
            additional_args_for_model["cache_dir"] = self.cache_dir

        additional_args_for_model.update(kwargs)
        return additional_args_for_model

    def create(self, **kwargs) -> PreTrainedModel:
        """
        Creates the model.

        :return: Model.
        """
        all_args = self.get_model_args(**kwargs)

        if self.config is not None:
            all_args.update(self.config)
            config, unused_kwargs = self.CONFIG_LOADER.from_pretrained(
                self.model_path,
                return_unused_kwargs=True,
                trust_remote_code=self.trust_remote_code,
                **all_args
            )
            if self.random_initialization:
                return self.LOADER.from_config(config, **unused_kwargs, trust_remote_code=self.trust_remote_code)
            return self.LOADER.from_pretrained(self.model_path, config=config, trust_remote_code=self.trust_remote_code, **unused_kwargs)

        if self.random_initialization:
            config, unused_kwargs = self.CONFIG_LOADER.from_pretrained(self.model_path, return_unused_kwargs=True, trust_remote_code=self.trust_remote_code, **all_args)
            return self.LOADER.from_config(config, trust_remote_code=self.trust_remote_code, **unused_kwargs)
        return self.LOADER.from_pretrained(self.model_path, trust_remote_code=self.trust_remote_code, **all_args)


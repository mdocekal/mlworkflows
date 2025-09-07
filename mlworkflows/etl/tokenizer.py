from typing import Optional
from classconfig import ConfigurableMixin, ConfigurableValue, RelativePathTransformer
from classconfig.validators import BoolValidator, ListOfTypesValidator
from transformers import PreTrainedTokenizerBase, AutoTokenizer


class TokenizerFactory(ConfigurableMixin):
    """
    Factory for transformers tokenizer.
    """

    name_or_path: str = ConfigurableValue("Name or path to the tokenizer.",
                                          transform=RelativePathTransformer(force_relative_prefix=True))
    use_fast: bool = ConfigurableValue("Use fast tokenizer if it is supported.", user_default=True,
                                       validator=BoolValidator(),
                                       voluntary=True)
    cache_dir: Optional[str] = ConfigurableValue("Cache directory.",
                                                 user_default=None,
                                                 transform=RelativePathTransformer(allow_none=True),
                                                    voluntary=True)

    add_tokens_to_vocab: list[str] = ConfigurableValue("Add tokens to the vocabulary.", user_default=[],
                                                                validator=ListOfTypesValidator(
                                                                    str, allow_empty=True
                                                                ),
                                                                voluntary=True)
    shared: bool = ConfigurableValue("True if the tokenizer should be created only once and then reused for each create call.",
                                     user_default=False, validator=BoolValidator(), voluntary=True)
    max_length: Optional[int] = ConfigurableValue("Set/override the maximum length of the model’s input sequences.",
                                                    user_default=None, voluntary=True)
    trust_remote_code: Optional[bool] = ConfigurableValue(
        "Whether to trust remote code.",
        user_default=False,
        voluntary=True
    )

    def __post_init__(self):
        self._shared_tokenizer = None

    def create(self) -> PreTrainedTokenizerBase:
        """
        Creates the tokenizer.

        :return: Tokenizer.
        """
        if self.shared and self._shared_tokenizer is not None:
            return self._shared_tokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.name_or_path, use_fast=self.use_fast,
                                                  cache_dir=self.cache_dir,
                                                  trust_remote_code=self.trust_remote_code)

        if self.max_length is not None:
            tokenizer.model_max_length = self.max_length

        if len(self.add_tokens_to_vocab) > 0:
            tokenizer.add_tokens(self.add_tokens_to_vocab)

        if self.shared:
            self._shared_tokenizer = tokenizer

        return tokenizer


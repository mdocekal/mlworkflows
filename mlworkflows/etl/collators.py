import itertools
from dataclasses import dataclass
from typing import List, Optional, Union, Iterable

import numpy
import torch
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin, DataCollatorWithPadding
from transformers.utils import PaddingStrategy

from mlworkflows.etl.batches import StructIndicesFactory
from mlworkflows.etl.transform import masked_language_modeling_masks, mlm_transform


@dataclass
class BatchIndexedDataCollatorForTokenClassification(DataCollatorMixin):
    """
    Data collator for token classification that takes a list of features and convert them into a indexed batch.
    """

    return_tensors: str = "pt"

    def __init__(self, make_matrix_compatible: bool = False):
        """

        :param make_matrix_compatible: There were some compatibility issues with the HF Trainer, so it is possible
        to convert the batch vectors to 1xN matrices.
        """
        self.make_matrix_compatible = make_matrix_compatible

    def torch_call(self, features):
        import torch
        assert "batch_index" not in features[0].keys(), "Features already have a batch index."

        # create batch index
        batch_index = itertools.chain.from_iterable([i] * len(feature["input_ids"]) for i, feature in enumerate(features))
        batch_index = torch.tensor(list(batch_index), dtype=torch.long)

        label_name = "label" if "label" in features[0].keys() else "labels"

        batch = {
            "batch_index": batch_index,
        }

        if label_name in features[0].keys():
            batch[label_name] = torch.hstack([
                torch.tensor(feature[label_name], dtype=torch.long) for feature in features
            ])

        for k, v in features[0].items():
            if k != label_name and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.hstack([f[k] for f in features])
                elif isinstance(v, numpy.ndarray):
                    batch[k] = torch.from_numpy(numpy.hstack([f[k] for f in features]))
                elif isinstance(v, Iterable):
                    dtype = None
                    if k == "input_ids":
                        dtype = torch.long

                    batch[k] = torch.tensor(list(itertools.chain.from_iterable(f[k] for f in features)), dtype=dtype)
                else:
                    dtype = None
                    if isinstance(v, int):
                        dtype = torch.long
                    batch[k] = torch.tensor(list(f[k] for f in features), dtype=dtype)

        if self.make_matrix_compatible:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.unsqueeze(0)
        return batch

    def tf_call(self, features):
        raise NotImplementedError("TensorFlow is not supported.")

    def numpy_call(self, features):
        raise NotImplementedError("NumPy is not supported.")


@dataclass
class StructDataCollatorForTokenClassification(BatchIndexedDataCollatorForTokenClassification):
    """
    Data collator for token classification that takes a list of features and convert them into strct batch.
    """

    def __init__(self, pursuit_minimal_context: bool, levels: List[int], make_matrix_compatible: bool = False):
        """
        :param pursuit_minimal_context: If True then the context size for each level is minimal for given input.
        :param levels: Defines context sizes for each level and number of levels.
        :param make_matrix_compatible: There were some compatibility issues with the HF Trainer, so it is possible
        to convert the batch vectors to 1xN matrices.
        """
        super().__init__(make_matrix_compatible=make_matrix_compatible)
        self.indices_factory = StructIndicesFactory(
            pursuit_minimal_context=pursuit_minimal_context,
            levels=levels
        )

    def torch_call(self, features):
        batch = super().torch_call(features)
        batch["indices"] = self.indices_factory.create_indices(batch["batch_index"].flatten())
        del batch["batch_index"]

        return batch


@dataclass
class MaskedLanguageModelingDataCollator(DataCollatorWithPadding, DataCollatorMixin):
    """
    Data collator for masked language modeling.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase,
                 padding: Union[bool, str, PaddingStrategy] = True,
                 max_length: Optional[int] = None,
                 pad_to_multiple_of: Optional[int] = None,
                 selection_prob: float = 0.1,
                 span_selection_prob: float = 0.0, masking_prob: float = 0.8, random_prob: float = 0.1,
                 ignore: int = -100, span_sep_token: Optional[int] = None,
                 do_not_select_pad: bool = True):
        """
        :param tokenizer: Tokenizer that will be to decode the input to obtain the gt sequence.
        :param padding: Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        :param max_length: Maximum length of the returned list and optionally padding length (see above).
        :param pad_to_multiple_of: If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.0 (Volta).
        :param selection_prob: Probability that given token will be selected for masked language modeling.
        :param span_selection_prob: Probability that given span will be selected for masked language modeling.
            It will select a token with id `span_sep_token` with this probability and then it will select
            all tokens between this token and the next `span_sep_token` token or the end of the sequence.
            You must set `span_sep_token` if you want to use this feature.
        :param masking_prob: Probability that selected token will be replaced with MASK token.
        :param random_prob: Probability that selected token will be replaced with random token.
        :param ignore: ignore label
        :param span_sep_token: token that will be used to separate spans
        :param do_not_select_pad: If True then [PAD] tokens will not be selected for masking.
            This is useful for special tokens like [PAD] that should not be masked.
        :raises ValueError: If `span_selection_prob` is greater than 0.0 and `span_sep_token` is None.
        """
        super().__init__(tokenizer=tokenizer, padding=padding, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of)
        self.selection_prob = selection_prob
        self.span_selection_prob = span_selection_prob
        self.masking_prob = masking_prob
        self.random_prob = random_prob
        self.ignore = ignore
        self.span_sep_token = span_sep_token
        self.do_not_select_pad = do_not_select_pad

        if span_selection_prob > 0.0 and span_sep_token is None:
            raise ValueError("You must set `span_sep_token` if you want to use `span_selection_prob`.")

        self.unselectable_token_id = tokenizer.pad_token_id if do_not_select_pad else None

    def torch_call(self, features):
        features = super()(features)
        mlm_transform(
            features=features,
            selection_prob=self.selection_prob,
            span_selection_prob=self.span_selection_prob,
            masking_prob=self.masking_prob,
            random_prob=self.random_prob,
            vocab_size=self.tokenizer.vocab_size,
            mask_token_id=self.tokenizer.mask_token_id,
            span_sep_token=self.span_sep_token,
            ignore=self.ignore,
            unselectable_token_id=self.unselectable_token_id
        )

        return features


class StructMaskedLanguageModelingDataCollator(BatchIndexedDataCollatorForTokenClassification):
    """
    Data collator for masked language modeling that creates a structured batch.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, levels: List[int], pursuit_minimal_context: bool = False,
                 make_matrix_compatible: bool = False, selection_prob: float = 0.1,
                 span_selection_prob: float = 0.0, masking_prob: float = 0.8, random_prob: float = 0.1,
                 ignore: int = -100, span_sep_token: Optional[int] = None):
        """
        :param tokenizer: Tokenizer that will be to decode the input to obtain the gt sequence.
        :param levels: Defines context sizes for each level and number of levels.
        :param pursuit_minimal_context: If True then the context size for each level is minimal for given input.
        :param make_matrix_compatible: There were some compatibility issues with the HF Trainer, so it is possible
            to convert the batch vectors to 1xN matrices.
        :param selection_prob: Probability that given token will be selected for masked language modeling.
        :param masking_prob: Probability that selected token will be replaced with MASK token.
        :param random_prob: Probability that selected token will be replaced with random token.
        :param span_selection_prob: Probability that given span will be selected for masked language modeling.
            It will select a token with id `span_sep_token` with this probability and then it will select
            all tokens between this token and the next `span_sep_token` token or the end of the sequence.
            You must set `span_sep_token` if you want to use this feature.
        :param ignore: ignore label
        :param span_sep_token: token that will be used to separate spans
        :raises ValueError: If `span_selection_prob` is greater than 0.0 and `span_sep_token` is None.
        """
        super().__init__(make_matrix_compatible)
        self.tokenizer = tokenizer
        self.indices_factory = StructIndicesFactory(
            pursuit_minimal_context=pursuit_minimal_context,
            levels=levels
        )

        self.selection_prob = selection_prob
        self.span_selection_prob = span_selection_prob
        self.masking_prob = masking_prob
        self.random_prob = random_prob
        self.ignore = ignore
        self.span_sep_token = span_sep_token

        if span_selection_prob > 0.0 and span_sep_token is None:
            raise ValueError("You must set `span_sep_token` if you want to use `span_selection_prob`.")

    def torch_call(self, features):
        features = super().torch_call(features)
        features["indices"] = self.indices_factory.create_indices(features["batch_index"].flatten())
        del features["batch_index"]

        mlm_transform(
            features=features,
            selection_prob=self.selection_prob,
            masking_prob=self.masking_prob,
            random_prob=self.random_prob,
            span_selection_prob=self.span_selection_prob,
            vocab_size=self.tokenizer.vocab_size,
            mask_token_id=self.tokenizer.mask_token_id,
            span_sep_token=self.span_sep_token,
            ignore=self.ignore
        )

        return features

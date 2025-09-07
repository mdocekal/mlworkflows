"""Module with utils for data transformation."""
from typing import Tuple, Optional

import torch


def masked_language_modeling_masks(input_ids: torch.Tensor, selection_prob: float = 0.1,
                                   span_selection_prob: float = 0.0, masking_prob: float = 0.8,
                                   random_prob: float = 0.1, span_sep_token: Optional[int] = None,
                                   unselectable_token_id: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Creates selection, masking, random masks for masked language modeling.

    :param input_ids: Input ids.
    :param selection_prob: Probability that given token will be selected for masked language modeling.
    :param span_selection_prob: Probability that given span will be selected for masked language modeling.
        It will select a token with id `span_sep_token` with this probability and then it will select
        all tokens between this token and the next `span_sep_token` token or the end of the sequence.
        You must set `span_sep_token` if you want to use this feature.
    :param masking_prob: Probability that selected token will be replaced with MASK token.
    :param random_prob: Probability that selected token will be replaced with random token.
    :param span_sep_token: token that will be used to separate spans
    :param unselectable_token_id: If provided, tokens with this ID will not be selected for masking.
        This is useful for special tokens like [PAD] that should not be masked.
    :return: selection mask, masking mask, random mask
    """

    if selection_prob > 0.0:
        selection_prob = torch.full(input_ids.shape, selection_prob)
        selected = torch.bernoulli(selection_prob).bool()
        if unselectable_token_id is not None:
            unselectable_mask = input_ids == unselectable_token_id
            selected[unselectable_mask] = False
    else:
        selected = torch.zeros(input_ids.shape, dtype=torch.bool)

    if span_selection_prob > 0.0:
        span_selection_prob = torch.full(input_ids.shape, span_selection_prob)
        span_selected = torch.bernoulli(span_selection_prob).bool()
        span_starts_mask = input_ids == span_sep_token
        span_starts_indices = torch.nonzero(span_starts_mask).flatten()
        span_selected[~span_starts_mask] = False
        selected_starts_indices = torch.nonzero(span_selected).flatten()

        span_starts_indices_offset = 0
        # fill the content of selected spans
        for i in selected_starts_indices + 1:
            while (span_starts_indices_offset < span_starts_indices.shape[0]
                   and span_starts_indices[span_starts_indices_offset] < i):
                span_starts_indices_offset += 1

            if span_starts_indices_offset == span_starts_indices.shape[0]:
                selected[i:] = True
            else:
                selected[i:span_starts_indices[span_starts_indices_offset]] = True

    masking_mask = torch.bernoulli(torch.full_like(input_ids, masking_prob, dtype=torch.float)).bool() & selected
    random_mask = torch.bernoulli(torch.full_like(input_ids, random_prob, dtype=torch.float)).bool() & selected & \
                  ~masking_mask

    return selected, masking_mask, random_mask


def mlm_transform(features: dict[str, torch.Tensor], selection_prob: float,
                  masking_prob: float,
                  random_prob: float,
                  span_selection_prob: float,
                  vocab_size: int,
                  mask_token_id: int,
                  span_sep_token: Optional[int] = None,
                  ignore: int = -100,
                  unselectable_token_id: Optional[int] = None) -> dict[str, torch.Tensor]:
    """
    Applies masked language modeling transformation to the input features.

    It will change the input_ids in-place and create labels.

    :param features: Input features (input_ids).
    :param selection_prob: Probability that given token will be selected for masked language modeling.
    :param masking_prob: Probability that selected token will be replaced with MASK token.
    :param random_prob: Probability that selected token will be replaced with random token.
    :param span_selection_prob: Probability that given span will be selected for masked language modeling.
        It will select a token with id `span_sep_token` with this probability and then it will select
        all tokens between this token and the next `span_sep_token` token or the end of the sequence.
        You must set `span_sep_token` if you want to use this feature.
    :param vocab_size: Size of the vocabulary, used for random token replacement.
    :param mask_token_id: ID of the token that will be used for masking.
    :param span_sep_token: token that will be used to separate spans
    :param ignore: Value to use for labels where no masking is applied (default is -100).
    :param unselectable_token_id: If provided, tokens with this ID will not be selected for masking.
        This is useful for special tokens like [PAD] that should not be masked.
    :return: Transformed features with masked input_ids and corresponding labels.
    """

    selected, masking_mask, random_mask = masked_language_modeling_masks(
        features["input_ids"], selection_prob=selection_prob, span_selection_prob=span_selection_prob,
        masking_prob=masking_prob, random_prob=random_prob, span_sep_token=span_sep_token,
        unselectable_token_id=unselectable_token_id
    )
    features["labels"] = features["input_ids"].clone()
    features["labels"][~selected] = ignore
    # we do not want to calc loss on unselected so we use the ignore_index

    if masking_prob > 0.0:
        # Replace with mask token
        features["input_ids"][masking_mask] = mask_token_id

    if random_prob > 0.0:
        # Replace with random token
        features["input_ids"][random_mask] = torch.randint(
            vocab_size, (random_mask.sum().item(),), dtype=torch.long
        )

    return features

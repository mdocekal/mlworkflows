from typing import Sequence


def align_tokens(tokens: Sequence[tuple[int, int]], align_to: Sequence[tuple[int, int]]) -> list[int]:
    """
    Aligns tokens to the target sequence.
    It is expected that the tokens are sorted by the start offset, and they don't overlap.

    :param tokens: Tokens to align in form of (start, end) offsets.
    :param align_to: Sequence to align to in form of (start, end) offsets.
    :return: Aligned index for each token.
    """

    aligned = []
    align_to_idx = 0
    already_aligned = False  # only the first token that aligns to the target word/token is aligned
    for token in tokens:
        if token[0] == 0 and token[1] == 0:
            # special token
            aligned.append(-1)
            continue

        while align_to_idx < len(align_to) and align_to[align_to_idx][1] <= token[0]:
            already_aligned = False
            align_to_idx += 1

        if align_to_idx >= len(align_to):
            aligned.append(-1)
            continue

        if token[1] > align_to[align_to_idx][0] and token[0] < align_to[align_to_idx][1]:
            if already_aligned:
                aligned.append(-1)
            else:
                aligned.append(align_to_idx)
                already_aligned = True
        else:
            aligned.append(-1)

    return aligned


def assemble_token_offsets(tokens: Sequence[str], start_offsets: Sequence[int]) -> list[tuple[int, int]]:
    """
    Assembles token offsets from the tokens and start offsets.

    :param tokens: Tokens.
    :param start_offsets: Start offsets of the tokens.
    :return: List of token offsets in form of (start, end).
    """
    if len(tokens) != len(start_offsets):
        raise ValueError("Tokens and start offsets must have the same length.")

    offsets = []
    for token, start_offset in zip(tokens, start_offsets):
        offsets.append((start_offset, start_offset + len(token)))

    return offsets

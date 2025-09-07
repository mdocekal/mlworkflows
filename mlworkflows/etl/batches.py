import math
from dataclasses import dataclass
from typing import Tuple, List, Sequence

import torch
from torch_scatter import scatter


@dataclass
class StructIndices:
    """
    Structure that contains indices used in struct model.
    """
    batch_index: torch.Tensor  # batch index for the original input sequence
    neighbourhoods: Sequence[torch.Tensor]   # mask for each parent neighbourhood, for each level it index of associated parent node
    original_batch_sample_identifier: Sequence[torch.Tensor] # For each tree node it stores the index of the original batch sample it is associated with
    expand_lvl_batch_index: Sequence[torch.Tensor]  # for each lvl it contains batch index mask for expanding parents to all associated leafs (no indices for leaf levels)
    position_ids: Sequence[torch.Tensor]  # position ids for each level, if None then not used

    def to(self, *args, **kwargs):
        """
        Moves all tensors in the indices to the given device.

        :return: StructIndices with tensors moved to the given device
        """
        return StructIndices(
            batch_index=self.batch_index.to(*args, **kwargs),
            neighbourhoods=[b.to(*args, **kwargs) for b in self.neighbourhoods],
            original_batch_sample_identifier=[b.to(*args, **kwargs) for b in self.original_batch_sample_identifier],
            expand_lvl_batch_index=[b.to(*args, **kwargs) for b in self.expand_lvl_batch_index],
            position_ids=[p.to(*args, **kwargs) for p in self.position_ids]
        )

    def pin_memory(self) -> "StructIndices":
        """
        Pins all tensors in the indices to the pinned memory.

        :return: StructIndices with tensors pinned to the pinned memory
        """
        self.batch_index = self.batch_index.pin_memory()
        self.neighbourhoods = [b.pin_memory() for b in self.neighbourhoods]
        self.original_batch_sample_identifier = [b.pin_memory() for b in self.original_batch_sample_identifier]
        self.expand_lvl_batch_index = [b.pin_memory() for b in self.expand_lvl_batch_index]
        self.position_ids = [p.pin_memory() for p in self.position_ids]

        return self


class StructIndicesFactory:
    def __init__(self, pursuit_minimal_context: bool, levels: List[int]):
        """
        :param pursuit_minimal_context: If True then the context size for each level is minimal for given input.
        :param levels: Defines context sizes for each level and number of levels.
        """
        self._levels = levels
        self._pursuit_minimal_context = pursuit_minimal_context
        self.lvl_max_context_sizes = torch.tensor(levels, dtype=torch.long)

        # let's create a structure that is useful for calculating segment length for given level
        # for each lvl we will compute max context size that can be accepted by all the levels above
        # for example if we have 3 levels with context sizes 2, 4, 8 then the max context sizes are:
        #   0 -> 4*8 = 32
        #   1 -> 8
        self.cum_context_sizes = torch.cumprod(self.lvl_max_context_sizes.flip(0), dim=0).flip(0)[1:]

    def calc_segment_len(self, sequence_length: int, lvl: int) -> int:
        """
        Calculates segment length for given level given seguence length of the input sequence.

        :param sequence_length: Length of the input sequence.
        :param lvl: pooling level
        :return: segment length
        """

        current_segment_len = self.lvl_max_context_sizes[lvl]
        if self._pursuit_minimal_context and lvl == 0:
            current_segment_len = max(2, math.ceil(sequence_length / self.cum_context_sizes[lvl]))

        current_segment_len = min(sequence_length, current_segment_len)

        return current_segment_len

    @staticmethod
    def prepare_batch_index(batch_index: torch.Tensor, current_segment_len: torch.Tensor) -> torch.Tensor:
        """
        Transforms batch index to the current level.

        it will split each previous batch into batches with current segment length
        example: previous batch [0,0,0,0, 1,1,1,1, 2,2,2,2], current segment length 3
            -> [0,0,0,1, 2,2,2,3, 4,4,4,5]

        :param batch_index: previous batch index
            [batch_size*previous_seq_len]
        :param current_segment_len: current segment length for each batch sample
        :return: batch index for current level
        """
        _, inverse_indices, counts = torch.unique(batch_index, return_inverse=True, return_counts=True)

        sub_batches_per_batch = torch.ceil(counts.double() / current_segment_len).long()
        # -> [batch_size] -> [2,2,2]
        sub_batches_start_offsets = torch.zeros_like(counts)
        # -> [batch_size] -> [0,0,0]
        sub_batches_start_offsets[1:] = torch.cumsum(sub_batches_per_batch[:-1], dim=0)
        # -> [batch_size] -> [0,2,4]

        cum_counter = torch.zeros_like(counts)
        # -> [batch_size] -> [0,0,0]
        cum_counter[1:] = torch.cumsum(counts[:-1], dim=0)
        # -> [batch_size] -> [0,4,8]

        res = torch.arange(batch_index.shape[0], dtype=torch.long, device=batch_index.device)
        # -> [batch_size*previous_seq_len] -> [0,1,2,3, 4,5,6,7, 8,9,10,11]
        res = res - cum_counter[inverse_indices]
        # -> [0,1,2,3, 0,1,2,3, 0,1,2,3]
        res = res // current_segment_len[inverse_indices]
        # -> [0,1,2,3, 0,1,2,3, 0,1,2,3] -> [0,0,0,1, 0,0,0,1, 0,0,0,1]
        res = res + sub_batches_start_offsets[inverse_indices]
        # -> [0,0,0,1,1,1,2,2,2,3,3,3] -> [0,0,0,1, 2,2,2,3, 4,4,4,5]

        return res

    @staticmethod
    def prepare_position_ids(batch_index: torch.Tensor) -> torch.Tensor:
        """
        Assigns position ids for given batch index.

        example:  [0,0,0,1,2,2,2,3,4,4,4,5]
            -> [0,1,2,0,0,1,2,0,0,1,2,0]


        :param batch_index: previous batch index
            [batch_size*previous_seq_len]
        :return: position ids
        """
        _, inverse_indices, counts = torch.unique(batch_index, return_inverse=True, return_counts=True)
        cum_counter = torch.zeros_like(counts)
        # -> [batch_size] -> [0,0,0,0,0,0]
        cum_counter[1:] = torch.cumsum(counts[:-1], dim=0)
        # -> [batch_size] -> [0,3,4,7,8,11]

        res = torch.arange(batch_index.shape[0], dtype=torch.long, device=batch_index.device)
        # -> [batch_size*previous_seq_len] -> [0,1,2,3, 4,5,6,7, 8,9,10,11]
        res = res - cum_counter[inverse_indices]
        # -> [0,1,2, 0, 0,1,2, 0, 0,1,2, 0]
        return res

    @staticmethod
    def prepare_batch_index_for_next_lvl(batch_index_previous: torch.Tensor, current_batch_index: torch.Tensor) \
            -> torch.Tensor:
        """
        Creates batch index for next level

        example: previous batch [0,0,0,0,1,1,1,1,2,2,2,2], current_batch_index [0,0,0,1,2,2,2,3,4,4,4,5]
            -> [0,0,1,1,2,2]

        :param batch_index_previous: batch index from previous level
        :param current_batch_index: batch index for current level, that is used for separating sample into
        smaller context sizes
        :return: batch index for next level
        """

        _, counts = torch.unique(current_batch_index, return_counts=True)
        cur_batch_offsets = torch.zeros_like(counts)
        # -> [batch_size] -> [0,0,0,0,0,0]
        cur_batch_offsets[1:] = torch.cumsum(counts[:-1], dim=0)
        # -> [batch_size] -> [0,3,4,7,8,11]

        return batch_index_previous[cur_batch_offsets]

    @staticmethod
    def tree_lvl_groups(current_batch_index: torch.Tensor) -> torch.Tensor:
        """
        Creates groups of indices for each tree level.

        E.g.
            current_batch_index: [0,0,0,1,2,2,2,3,4,4,4,5]
            -> [[0,1,2], [3], [4,5,6], [7], [8,9,10], [11]]
            -> [[0,3], [3,4], [4,7], [7,8], [8,11], [11,12]]

        :param current_batch_index: batch index for current level
            [all_tokens_in_lvl]
        :return: groups of indices for each level
            groups are represented by indices of the first and last element (+1) of the group
        """

        lvl_indices = torch.arange(len(current_batch_index), dtype=torch.long, device=current_batch_index.device)
        min_per_group = scatter(lvl_indices, current_batch_index, dim=0, reduce="min")
        max_per_group = scatter(lvl_indices, current_batch_index, dim=0, reduce="max")
        return torch.stack([min_per_group, max_per_group + 1], dim=1)

    @torch.no_grad()
    def prepare_multi_lvl_batch_index(self, batch_index: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Prepares batch index for each level.

        :param batch_index: batch index for current level
            [all_tokens]
        :return: batch index for each level
            original batch sample identifier index
            expanded batch index for each level
        """

        multi_lvl_batch_index = [batch_index.detach().clone()]
        original_batch_sample_identifier = [batch_index.detach().clone()]
        expand_lvl_batch_index = []
        current_batch_index = batch_index
        _, current_segment_len = torch.unique(current_batch_index, return_counts=True)

        for lvl in range(len(self._levels)):
            stored_batch_index = current_batch_index.detach().clone()  # we will need previous batch index for next level

            current_segment_len = [self.calc_segment_len(seq_len.item(), lvl) for seq_len in current_segment_len]
            current_segment_len = torch.tensor(current_segment_len, dtype=torch.long, device=batch_index.device)

            # we will split each previous batch into batches with current segment length
            # example: previous batch [0,0,0,0,1,1,1,1,2,2,2,2], segment length 3
            #   -> [0,0,0,1,2,2,2,3,4,4,4,5]
            current_batch_index = self.prepare_batch_index(current_batch_index, current_segment_len)
            detached_current_batch_index = current_batch_index.detach().clone()
            multi_lvl_batch_index.append(detached_current_batch_index)
            if len(expand_lvl_batch_index) == 0:
                expand_lvl_batch_index.append(detached_current_batch_index)
            else:
                expand_lvl_batch_index.append(detached_current_batch_index[expand_lvl_batch_index[-1]])

            # we need to update the batch index for the next level, so we will have states from same
            # batch sample together
            current_batch_index = self.prepare_batch_index_for_next_lvl(stored_batch_index, current_batch_index)

            original_batch_sample_identifier.append(current_batch_index.detach().clone())

        return multi_lvl_batch_index, original_batch_sample_identifier, expand_lvl_batch_index

    def create_indices(self, batch_index: torch.Tensor) -> StructIndices:
        """
        Creates indices for given batch index.

        :param batch_index: batch index for current level
            [all_tokens]
        :return: StructIndices
        """
        multi_lvl_batch_index, original_batch_sample_identifier, expand_lvl_batch_index = self.prepare_multi_lvl_batch_index(
            batch_index)
        return StructIndices(
            batch_index=multi_lvl_batch_index[0],  # the first one is the original batch index
            neighbourhoods=multi_lvl_batch_index[1:],  # skip the first one as it is the original batch index
            original_batch_sample_identifier=original_batch_sample_identifier,
            expand_lvl_batch_index=expand_lvl_batch_index,
            position_ids=[self.prepare_position_ids(b) for b in multi_lvl_batch_index[1:]]
            # skip the first one as it is the original batch index
        )


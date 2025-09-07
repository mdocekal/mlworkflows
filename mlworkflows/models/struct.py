# -*- coding: UTF-8 -*-

import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, List, Union, Tuple, Sequence

import torch
from pytorch_missing.indices_oriented import indices_weighted_scatter_sum, indices_dot_product
from pytorch_missing.ranges import mrange
from torch_geometric.utils import softmax
from torch_geometric.utils.scatter import scatter_argmax
from torch_scatter import scatter
from transformers import PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN

from suma.configuration.configurable import ConfigurableValue, MinValueIntegerValidator, StringValidator, \
    FloatValidator, AnyValidator, IsNoneValidator, ListOfTypesValidator, BoolValidator, ConfigurableMixin, \
    RelativePathTransformer, ConfigurableFactory
from suma.etl.batch import TorchTokenizedBatchIndexed, TorchTokenizedExtSumSegmentsClassificationBatchIndexed
from suma.etl.tokenizers import TransformersTokenizerFactory
from suma.models.base import HFLoadable
from suma.models.outputs import TorchSegOutput, ExtSummarizationResultOutput
from suma.models.trainable.torch.base import TorchModel


class StructConfig(PretrainedConfig):
    model_type = "struct"
    vocab_size: int = ConfigurableValue(
        "Vocabulary size of model.",
        user_default=8_000, validator=MinValueIntegerValidator(min_value=1))
    num_layers: int = ConfigurableValue("Number of layers in the model.",
                                        user_default=4, validator=MinValueIntegerValidator(min_value=1))
    mixing_size: int = ConfigurableValue("Size of the intermediate mixing feed-forward layer.",
                                            user_default=2048, validator=MinValueIntegerValidator(min_value=1))
    hidden_act: str = ConfigurableValue(
        "The non-linear activation function (function or string). If string, \"gelu\", "
        "\"relu\", \"silu\" and \"gelu_new\" are supported.",
        user_default="gelu", validator=StringValidator())
    hidden_dropout_prob: float = ConfigurableValue(
        "The dropout probability for all fully connected layers.",
        user_default=0.1, validator=FloatValidator(min_value=0.0, max_value=1.0))
    nary_search_dropout_prob: float = ConfigurableValue(
        "The dropout probability for the nary search.",
        user_default=0.1, validator=FloatValidator(min_value=0.0, max_value=1.0))
    init_std: float = ConfigurableValue(
        "The standard deviation of the truncated_normal_initializer for initializing all weight matrices.",
        user_default=0.02, validator=FloatValidator(min_value=0.0))
    layer_norm_eps: float = ConfigurableValue("The epsilon used by the layer normalization layers.",
                                              user_default=1e-12, validator=FloatValidator(min_value=0.0))
    levels: List[int] = ConfigurableValue(
        "Defines context sizes for each level and number of levels.",
        user_default=None, validator=ListOfTypesValidator(int, value_validator=MinValueIntegerValidator(min_value=1)), )
    hier_level_embedding_size: Union[int, List[int]] = ConfigurableValue(
        "Embedding size for hierarchical levels. You can use single value for all levels or list of values for each level. If passing list do not forget to pass the zeroth level which is the size for embeddings in vocabulary. Passing list will cause that the positional embeddings will have separate embedding matrix for each level.",
        user_default=256, validator=AnyValidator([MinValueIntegerValidator(min_value=1), ListOfTypesValidator(int, value_validator=MinValueIntegerValidator(min_value=1))]))
    lvl_reduction: str = ConfigurableValue(
        "How to reduce embeddings from previous level to the current level. Available options are: sum, mean, max.",
        user_default="sum", validator=StringValidator())

    pursuit_minimal_context: bool = ConfigurableValue(
        "If True then the context size for each level is minimal for given input. ", user_default=False,
        validator=BoolValidator())

    n_ary_search_dim: int = ConfigurableValue("Number of dimensions for n-ary search embeddings.",
                                              user_default=128,
                                              validator=MinValueIntegerValidator(min_value=1))
    num_sources: int = ConfigurableValue("Number of sources for n-ary search.",
                                            user_default=1, validator=MinValueIntegerValidator(min_value=1))
    num_heads: int = ConfigurableValue("Number of n-ary search heads.",
                                        user_default=1, validator=MinValueIntegerValidator(min_value=1))

    def __init__(self,
                 vocab_size: int = 8_000,
                 num_layers: int = 4,
                 mixing_size: int = 2048,
                 hidden_act: str = "gelu",
                 hidden_dropout_prob: float = 0.1,
                 nary_search_dropout_prob: float = 0.1,
                 init_std: float = 0.02,
                 layer_norm_eps: float = 1e-12,
                 levels: List[int] = None,
                 hier_level_embedding_size: Union[int, List[int]] = 256,
                 lvl_reduction: str = "sum",
                 pursuit_minimal_context: bool = False,
                 n_ary_search_dim: int = 128,
                 num_sources: int = 1,
                 num_heads: int = 1,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.mixing_size = mixing_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.nary_search_dropout_prob = nary_search_dropout_prob
        self.init_std = init_std
        self.layer_norm_eps = layer_norm_eps
        self.levels = levels
        self.hier_level_embedding_size = hier_level_embedding_size
        self.lvl_reduction = lvl_reduction
        self.pursuit_minimal_context = pursuit_minimal_context
        self.n_ary_search_dim = n_ary_search_dim
        self.num_sources = num_sources
        self.num_heads = num_heads

    @property
    def max_position_embeddings(self):
        return max(self.levels)

    def embedding_size_for_level(self, level: int) -> int:
        """
        Returns embedding size for given level.

        :param level: pooling level
            The zeroth level is for vocabulary embeddings.
        :return: embedding size after n-th pooling
        """
        if self.is_homogenous_embedding_size():
            return self.hier_level_embedding_size
        return self.hier_level_embedding_size[level]

    @property
    def hier_embedding_size(self):
        return self.embedding_size_for_level(0) + sum(self.embedding_size_for_level(1+i) for i in range(len(self.levels)))

    def is_homogenous_embedding_size(self) -> bool:
        return isinstance(self.hier_level_embedding_size, int)

    def is_homogenous_context_size(self) -> bool:
        return all(self.levels[0] == lvl for lvl in self.levels)

    def is_homogenous(self) -> bool:
        """
        Checks if the configuration is homogenous.
        """
        return self.is_homogenous_embedding_size() and self.is_homogenous_context_size()


class StructPositionalEncoding(torch.nn.Module):
    def __init__(self, config: StructConfig):
        """
        :param config: configuration
        """
        super().__init__()

        self.config = config
        self.position_embeddings = None

        if isinstance(config.hier_level_embedding_size, int):
            self.position_embeddings = torch.nn.Embedding(config.max_position_embeddings, config.hier_level_embedding_size)
        else:
            self.position_embeddings_per_level = torch.nn.ModuleList([
                torch.nn.Embedding(context_size, config.embedding_size_for_level(i))
                for i, context_size in range(len(config.levels))
            ])

    def forward(self, position_ids: torch.Tensor, level: int) -> torch.Tensor:
        """
        Computes embeddings from position ids.

        :param position_ids: Input ids.
        :param level: pooling level
        :return: Embeddings.
        """
        if self.position_embeddings is not None:
            return self.position_embeddings(position_ids)
        else:
            return self.position_embeddings_per_level[level](position_ids)


class StructVocabularyEmbeddings(torch.nn.Module):

    def __init__(self, config: StructConfig):
        super().__init__()

        self.embeddings = torch.nn.Embedding(config.vocab_size, config.embedding_size_for_level(0))
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Computes embeddings from input ids.

        :param input_ids: Input ids.
        :return: Embeddings.
        """
        embeddings = self.embeddings(input_ids)
        embeddings = self.dropout(embeddings)

        return embeddings


class StructLvl(torch.nn.Module):
    def __init__(self, config: StructConfig, inp_dim: int, out_dim: int):
        """
        :param config: configuration
        :param inp_dim: dimensions of input embeddings
        :param out_dim: dimensions of output embeddings
        """
        super().__init__()

        self.config = config
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(inp_dim, out_dim),
            ACT2FN[config.hidden_act]
        )

    def forward(self, prev_lvl_emb: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        """
        Computes representations for given level from previous level representations.

        :param prev_lvl_emb: embeddings from previous level
            [prev_embeddings, prev_lvl_emb_size]
        :param batch_index: batch index for current level
        :return: context embeddings for each batch sample
            [batch_size, curr_lvl_emb_size]
        """

        to_aggregate = self.projection(prev_lvl_emb)
        # -> [prev_embeddings, curr_lvl_emb_size]

        return scatter(to_aggregate, batch_index, dim=0, reduce=self.config.lvl_reduction)
        # -> [batch_size, curr_lvl_emb_size]


class StructHierPooling(torch.nn.Module):
    def __init__(self, config: StructConfig, position_embeddings: Optional[StructPositionalEncoding] = None,
                 vocab_input: bool = False, for_nary_search: bool = False):
        """
        :param config: configuration of model
        :param position_embeddings: embeddings for position ids
        :param vocab_input: if True then it is expected that the input consists of vocabulary embeddings
        :param for_nary_search: if True then the per lvl embeddings dimension will be of size n_ary_search_dim
        """

        super().__init__()
        self.config = config
        self.for_nary_search = for_nary_search

        self.position_embeddings = position_embeddings

        self.vocab_input = vocab_input

        self.levels = torch.nn.ModuleList([
            StructLvl(
                config,
                self.get_embedding_size(i),
                self.get_embedding_size(i+1)
            ) for i in range(len(config.levels))
        ])

        if self.position_embeddings is not None:
            self.layer_norm_after_pos_embeddings = torch.nn.ModuleList([
                torch.nn.LayerNorm(self.get_embedding_size(i),
                                   eps=config.layer_norm_eps)
                for i in range(len(config.levels))
            ])

        if not vocab_input:
            self.input_projection = torch.nn.Sequential(
                torch.nn.Linear(config.hier_embedding_size, self.get_embedding_size(0)),
                ACT2FN[config.hidden_act]
            )

        self.lvl_max_context_sizes = torch.tensor(config.levels, dtype=torch.long)

        # let's create a structure that is useful for calculating segment length for given level
        # for each lvl we will compute max context size that can be accepted by all the levels above
        # for example if we have 3 levels with context sizes 2, 4, 8 then the max context sizes are:
        #   0 -> 4*8 = 32
        #   1 -> 8
        self.cum_context_sizes = torch.cumprod(self.lvl_max_context_sizes.flip(0), dim=0).flip(0)[1:]

        self.pursuit_minimal_context = config.pursuit_minimal_context

    def get_embedding_size(self, lvl: int) -> int:
        """
        Returns embedding size for given level.

        :param lvl: level
        :return: embedding size
        """
        if self.vocab_input and lvl == 0:
            return self.config.embedding_size_for_level(0)

        return self.config.n_ary_search_dim if self.for_nary_search else self.config.embedding_size_for_level(lvl)

    def calc_segment_len(self, sequence_length: int, lvl: int) -> int:
        """
        Calculates segment length for given level given seguence length of the input sequence.

        :param sequence_length: Length of the input sequence.
        :param lvl: pooling level
        :return: segment length
        """

        current_segment_len = self.lvl_max_context_sizes[lvl]
        if self.pursuit_minimal_context and lvl == 0:
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
    def prepare_multi_lvl_batch_index(self, batch_index: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Prepares batch index for each level.

        :param batch_index: batch index for current level
            [all_tokens]
        :return: batch index for each level
            original batch sample identifier index
        """

        multi_lvl_batch_index = [batch_index.detach().clone()]
        original_batch_sample_identifier = [batch_index.detach().clone()]
        current_batch_index = batch_index
        _, current_segment_len = torch.unique(current_batch_index, return_counts=True)

        for lvl in range(len(self.config.levels)):
            stored_batch_index = current_batch_index.detach().clone()  # we will need previous batch index for next level

            current_segment_len = [self.calc_segment_len(seq_len.item(), lvl) for seq_len in current_segment_len]
            current_segment_len = torch.tensor(current_segment_len, dtype=torch.long, device=batch_index.device)

            # we will split each previous batch into batches with current segment length
            # example: previous batch [0,0,0,0,1,1,1,1,2,2,2,2], segment length 3
            #   -> [0,0,0,1,2,2,2,3,4,4,4,5]
            current_batch_index = self.prepare_batch_index(current_batch_index, current_segment_len)
            multi_lvl_batch_index.append(current_batch_index.detach().clone())

            # we need to update the batch index for the next level, so we will have states from same
            # batch sample together
            current_batch_index = self.prepare_batch_index_for_next_lvl(stored_batch_index, current_batch_index)

            original_batch_sample_identifier.append(current_batch_index.detach().clone())

        return multi_lvl_batch_index, original_batch_sample_identifier

    @torch.no_grad()
    def rightmost_node_children_count(self, batch_index: torch.Tensor) -> List[torch.Tensor]:
        """
        Computes number of children for rightmost node of each level for each batch sample.

        :param batch_index: batch index
            [all_tokens]
        :return: number of children for rightmost node of each level for each batch sample
        """

        original_batch_sample_identifier = self.prepare_multi_lvl_batch_index(batch_index)[1]
        res = []

        for lvl, for_lvl in enumerate(original_batch_sample_identifier[:-1]):   # we do not need the last level
            cnts = torch.unique(for_lvl, return_counts=True)[1]
            n_childs = cnts % self.lvl_max_context_sizes[lvl]
            n_childs[n_childs == 0] = self.lvl_max_context_sizes[lvl]
            res.append(n_childs)

        return res

    def forward(self, hidden_states: torch.Tensor, multi_lvl_batch_index: Sequence[torch.Tensor],
                flattened: bool = True,
                original_batch_sample_identifier: Optional[Sequence[torch.Tensor]] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]]:
        """
        Computes output of the layer.

        :param hidden_states: hidden states from previous layer
            [all_tokens, input_dim]
        :param multi_lvl_batch_index: batch index for each level
        :param flattened: if True then the output will be flattened
        :param original_batch_sample_identifier: only used when flattened is False
            For each tree node it stores the index of the original batch sample it is associated with
        :return:
            flattened True:
                 matrix of embeddings for each token
                 [all_tokens, hier_embedding_size]
            flattened False:
                Representation in form of tree. The tree is triplet of:
                    1. flattened embeddings
                    [all_tokens+tree nodes, dim]
                    2. tree index that stores indices of childs for each node in tree, leaf nodes are represented by empty interval (0,0)
                    3. indices to tree index storing root nodes for each batch sample
                        theoretically we may have multiple top level nodes for each batch sample (when the input is long)

                WARNING it is expecting that all levels have same number of dimensions for embeddings
        """
        level_states = hidden_states
        # -> [all_tokens, input_dim]

        if not self.vocab_input:
            level_states = self.input_projection(level_states)
            # -> [all_tokens, zero_lvl_emb_size]

        if flattened:
            res = [level_states]
        else:
            tree_embeddings = [level_states]
            tree_index = [
                torch.full((level_states.size(0), 2), 0, dtype=torch.long, device=level_states.device)
            ]  # leaf nodes
            number_of_nodes = level_states.size(0)

        for lvl, lvl_layer in enumerate(self.levels):
            current_batch_index = multi_lvl_batch_index[lvl + 1]  # +1 because 0 is the input batch index

            if self.position_embeddings is not None:
                position_ids = self.prepare_position_ids(current_batch_index)

                level_states = level_states + self.position_embeddings(position_ids, lvl)
                level_states = self.layer_norm_after_pos_embeddings[lvl](level_states)

            level_states = lvl_layer(level_states, current_batch_index)

            # -> [all_tokens / current_segment_len, pooling_lvl_size]

            if flattened:
                # expand the pooling states to the full length of the input sequence
                expanded_pooling_states = level_states
                for lvl_batch_index in reversed(multi_lvl_batch_index[1:lvl + 2]):
                    expanded_pooling_states = expanded_pooling_states.index_select(0, lvl_batch_index)
                # -> [all_tokens, pooling_size]

                res.append(expanded_pooling_states)
            else:
                tree_embeddings.append(level_states)
                tree_index.append(self.tree_lvl_groups(current_batch_index) + (number_of_nodes - tree_index[-1].shape[0]))
                number_of_nodes += tree_index[-1].size(0)

        if flattened:
            res = torch.cat(res, dim=-1)
        else:

            # we need to create virtual root nodes for each batch sample
            number_of_batch_samples = original_batch_sample_identifier[-1].max().item() + 1
            root_index = torch.arange(number_of_batch_samples, dtype=torch.long, device=tree_index[0].device) + number_of_nodes

            tree_index.append(
                self.tree_lvl_groups(original_batch_sample_identifier[-1]) + (number_of_nodes - tree_index[-1].shape[0])
            )
            tree_index = torch.cat(tree_index, dim=0)

            res = (
                torch.cat(tree_embeddings, dim=0),
                tree_index,
                root_index
            )

        return res


class StructMixingTransformation(torch.nn.Module):

    def __init__(self, config: StructConfig):
        """
        :param config: configuration
        """
        super().__init__()

        self.config = config

        self.to_intermediate = torch.nn.Linear(config.hier_embedding_size, config.mixing_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]
        self.dense = torch.nn.Linear(config.mixing_size, config.hier_embedding_size)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = torch.nn.LayerNorm(config.hier_embedding_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Computes output of the output transformation layer.

        :param hidden_states: Hidden states.
        :return: Output of the output transformation layer.
        """
        residual = hidden_states
        hidden_states = self.to_intermediate(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.layer_norm(hidden_states + residual)

        return hidden_states


class StructConstructionLayer(torch.nn.Module):
    def __init__(self, config: StructConfig, layer_num: int, position_embeddings: Optional[StructPositionalEncoding] = None):
        super().__init__()

        self.config = config
        self.layer_num = layer_num
        self.hier_pooling = StructHierPooling(config, position_embeddings, vocab_input=layer_num == 0)
        self.mixing = StructMixingTransformation(config)
        if layer_num != 0:
            self.pooling_norm = torch.nn.LayerNorm(config.hier_embedding_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, multi_lvl_batch_index: Sequence[torch.Tensor]) \
            -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Computes output of the shrinking layer.

        :param hidden_states: Hidden states.
        :param multi_lvl_batch_index: batch index for each level
        :return:
            representation of the input sequence
        """

        pooling_output = self.hier_pooling(hidden_states, multi_lvl_batch_index)

        if self.layer_num != 0:
            # residual connection, for first level it is not needed, as the input is part of hierarchical embeddings
            pooling_output = pooling_output + hidden_states
            pooling_output = self.pooling_norm(pooling_output)

        output = self.mixing(pooling_output)

        return output


class StructConstruction(torch.nn.Module):

    def __init__(self, config: StructConfig):
        super().__init__()

        self.config = config
        self.position_embeddings = StructPositionalEncoding(config)

        self.layers = torch.nn.ModuleList([
            StructConstructionLayer(config, i, self.position_embeddings if i == 0 else None)
            for i in range(config.num_layers)
        ])

    def forward(self, hidden_states: torch.Tensor, batch_index: torch.Tensor,
                multi_lvl_batch_index: Optional[Sequence[torch.Tensor]] = None) -> torch.Tensor:
        """
        Computes output of the body.

        :param hidden_states: Input hidden states.
        :param batch_index: Batch index.
        :param multi_lvl_batch_index: voluntary you can pass batch index for each level
            if None it will be obtained from batch_index
            WARNING: the batch_index is ignored when multi_lvl_batch_index is passed
        :return: representation of the input sequence
        """

        if multi_lvl_batch_index is None:
            multi_lvl_batch_index, _ = self.layers[0].hier_pooling.prepare_multi_lvl_batch_index(batch_index)

        for layer_i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, multi_lvl_batch_index)

        return hidden_states


class StructConstructionPreTrainedModel(PreTrainedModel):
    config_class = StructConfig
    base_model_prefix = "struct_construction"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class HFStructConstructionModel(StructConstructionPreTrainedModel):
    def __init__(self, config: StructConfig):
        super().__init__(config)
        self.vocabulary_embeddings = StructVocabularyEmbeddings(config)
        self.construction = StructConstruction(config)

        self.post_init()

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.vocabulary_embeddings.embeddings

    def set_input_embeddings(self, value: torch.nn.Module):
        self.vocabulary_embeddings.embeddings = value

    def forward(self, input_ids: torch.Tensor, batch_index: torch.Tensor,
                multi_lvl_batch_index: Optional[Sequence[torch.Tensor]] = None) -> torch.Tensor:
        """
        Computes output of the model.

        :param input_ids: Input ids.
        :param batch_index: Batch index.
        :param multi_lvl_batch_index: voluntary you can pass batch index for each level
            if None it will be obtained from batch_index
            WARNING: the batch_index is ignored when multi_lvl_batch_index is passed
        :return:
            representation of the input sequence
        """
        embedding_output = self.vocabulary_embeddings(input_ids)
        outputs = self.construction(embedding_output, batch_index, multi_lvl_batch_index)

        return outputs


class HFStructConstructionModelForTokenClassification(StructConstructionPreTrainedModel):
    def __init__(self, config: StructConfig):
        super().__init__(config)

        self.model = HFStructConstructionModel(config)

        self.lm_head = torch.nn.Linear(config.hier_embedding_size, config.num_labels, bias=False)
        self.config = config
        self.post_init()

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: torch.nn.Module):
        self.model.set_input_embeddings(value)

    def forward(self, input_ids: torch.Tensor, batch_index: torch.Tensor,
                prediction_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes output of the model.

        :param input_ids: Input ids.
        :param batch_index: Batch index.
        :param prediction_mask: If not None then it will be used as prediction mask.
            Prediction mask is used to filter tokens for which the lm_head will be computed.
        :return:
            prediction scores
        """
        outputs = self.model(input_ids, batch_index)

        pass_to_lm_head = outputs
        if prediction_mask is not None:
            pass_to_lm_head = pass_to_lm_head[prediction_mask]
        prediction_scores = self.lm_head(pass_to_lm_head)

        return prediction_scores


class HFStructConstructionModelWithMLMHead(HFStructConstructionModelForTokenClassification):
    """
    Same as HFStructConstructionModelForTokenClassification but it will set the vocab_size as num_labels.
    """

    def __init__(self, config: StructConfig):

        config.num_labels = config.vocab_size
        super().__init__(config)


class StructConstructionWithMLMHead(TorchModel, ConfigurableMixin, HFLoadable):
    """
    Struct construction model with language model head.
    """

    path_to_model: Optional[str] = ConfigurableValue("Path to the model. Use when you want to load a pretrained model. "
                                                     "When from scratch training is used use config instead.",
                                                     user_default="struct-construction-base",
                                                     transform=RelativePathTransformer(force_relative_prefix=True,
                                                                                       allow_none=True))
    config: StructConfig = ConfigurableFactory(
        StructConfig,
        "Configuration of the model. Use when you want to train from scratch. "
        "When pretrained model is used use path_to_model instead.",
        voluntary=True)
    tokenizer_factory: TransformersTokenizerFactory = ConfigurableFactory(TransformersTokenizerFactory,
                                                                          "Tokenizer configuration. It will use "
                                                                          "the tokenizer to obtain special tokens and "
                                                                          "to"
                                                                          "resize token embeddings if needed.")
    cache_dir: Optional[str] = ConfigurableValue("Path to cache directory.", user_default=None,
                                                 transform=RelativePathTransformer(allow_none=True))

    ignore_index: int = ConfigurableValue("Used to filter tokens for lm head, when gt tokens are provided.",
                                          user_default=-100)

    def __init__(self, **kwargs):
        TorchModel.__init__(self)
        ConfigurableMixin.__init__(self, **kwargs)

        if self.path_to_model is not None and isinstance(self.config, StructConfig):
            raise ValueError("Only one of path_to_model or config can be set.")

        if self.path_to_model is None and not isinstance(self.config, StructConfig):
            raise ValueError("One of path_to_model or config must be set.")

        if self.path_to_model is not None:
            self.model: HFStructConstructionModelWithMLMHead = HFStructConstructionModelWithMLMHead.from_pretrained(self.path_to_model,
                                                                                        cache_dir=self.cache_dir)
        else:
            self.model: HFStructConstructionModelWithMLMHead = HFStructConstructionModelWithMLMHead(self.config)

        self.tokenizer = self.tokenizer_factory.create()

        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, inputs: torch.Tensor, batch_index: torch.Tensor,
                prediction_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        """
        Forward pass of the model.

        :param inputs: Input ids.
        :param batch_index: Batch index.
        :param prediction_mask: If not None then it will be used as prediction mask.
            Prediction mask is used to filter tokens for which the lm_head will be computed.
        :return: Scores for each token.
        """
        return self.model(inputs, batch_index=batch_index, prediction_mask=prediction_mask)

    @staticmethod
    @torch.jit.script
    def predictions_to_batch(predictions: torch.Tensor, batch_index: torch.Tensor) -> List[List[int]]:
        """
        Converts predictions into batch list.

        :param predictions: token predictions
            [all_tokens]
        :param batch_index: batch index
            [all_tokens]
        :return: batch list
        """

        batch_list: List[List[int]] = []
        for i in torch.unique(batch_index):
            s: List[int] = predictions[batch_index == i].tolist()
            batch_list.append(s)
        return batch_list

    def __call__(self, batch: TorchTokenizedBatchIndexed) -> TorchSegOutput:
        prediction_mask = None
        if batch.gt_tokens is not None:
            prediction_mask = batch.gt_tokens != self.ignore_index

        out = self.forward(batch.inputs, batch.batch_index, prediction_mask=prediction_mask)

        predictions = None
        if not self.training:
            predictions = torch.argmax(out, dim=-1)
            final_predictions = batch.inputs.clone()
            if batch.gt_tokens is not None:
                final_predictions[prediction_mask] = predictions
            # convert predictions into batch list
            predictions = self.predictions_to_batch(final_predictions, batch.batch_index)
            predictions = self.tokenizer.batch_decode(predictions)

        return TorchSegOutput(sample_ids=batch.sample_ids,
                                        logits=out,
                                        sequences=predictions)


class ExtSumStructConstruction(TorchModel, ConfigurableMixin, HFLoadable):
    """
    Struct construction model with token classification head on top. The wrapper is used for extractive summarization.
    """

    path_to_model: Optional[str] = ConfigurableValue("Path to the model. Use when you want to load a pretrained model. "
                                                     "When from scratch training is used use config instead.",
                                                     user_default="struct-construction-base",
                                                     transform=RelativePathTransformer(force_relative_prefix=True,
                                                                                       allow_none=True))
    config: StructConfig = ConfigurableFactory(
        StructConfig,
        "Configuration of the model. Use when you want to train from scratch. "
        "When pretrained model is used use path_to_model instead.",
        voluntary=True)
    tokenizer_factory: TransformersTokenizerFactory = ConfigurableFactory(TransformersTokenizerFactory,
                                                                          "Tokenizer configuration. It will use "
                                                                          "the tokenizer to obtain special tokens and "
                                                                          "to"
                                                                          "resize token embeddings if needed.")
    gradient_checkpointing: bool = ConfigurableValue("Enable gradient checkpointing. ", user_default=False,
                                                     validator=BoolValidator())
    cache_dir: Optional[str] = ConfigurableValue("Path to cache directory.", user_default=None,
                                                 transform=RelativePathTransformer(allow_none=True))

    classification_threshold: float = ConfigurableValue("Threshold for classification.", user_default=0.5,
                                                        validator=FloatValidator(min_value=0.0, max_value=1.0))

    ignore_index: int = ConfigurableValue("Used to filter tokens for lm head, when gt tokens are provided.",
                                          user_default=-100)
    sequence_delimiter: str = ConfigurableValue("Delimiter for sequences.", user_default=" ",
                                                validator=AnyValidator([IsNoneValidator(), str]))

    def __init__(self, **kwargs):
        TorchModel.__init__(self)
        ConfigurableMixin.__init__(self, **kwargs)

        if self.path_to_model is not None and isinstance(self.config, StructConfig):
            raise ValueError("Only one of path_to_model or config can be set.")

        if self.path_to_model is None and not isinstance(self.config, StructConfig):
            raise ValueError("One of path_to_model or config must be set.")

        if self.path_to_model is not None:
            config = StructConfig.from_pretrained(self.path_to_model, cache_dir=self.cache_dir)
            config.num_labels = 1

            self.model: HFStructConstructionModelForTokenClassification = HFStructConstructionModelForTokenClassification.from_pretrained(
                self.path_to_model, config=config, cache_dir=self.cache_dir)
        else:
            self.config.num_labels = 1
            self.model: HFStructConstructionModelForTokenClassification = HFStructConstructionModelForTokenClassification(self.config)

        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.tokenizer = self.tokenizer_factory.create()

        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, inputs: torch.Tensor, batch_index: torch.Tensor,
                prediction_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        """
        Forward pass of the model.

        :param inputs: Input ids.
        :param batch_index: Batch index.
        :param prediction_mask: If not None then it will be used as prediction mask.
            Prediction mask is used to filter tokens for which the cls_head will be computed.
        :return: Scores for each token.
        """
        return self.model(inputs, batch_index=batch_index, prediction_mask=prediction_mask)

    def __call__(self, batch: TorchTokenizedExtSumSegmentsClassificationBatchIndexed) -> ExtSummarizationResultOutput:
        logits = self.forward(batch.inputs, batch.batch_index, prediction_mask=batch.classification_token_masks)[0]
        pred_sequences = None
        pred_classification = None
        if not self.training:
            predictions = logits.sigmoid()
            predictions = predictions > self.classification_threshold
            pred_sequences = []
            pred_classification = []
            segments_offset = 0
            flatten_segments = [item for sublist in batch.segments for item in sublist]

            for segments in batch.segments:
                sequence = []
                classification = []
                for _ in segments:
                    classification.append(int(predictions[segments_offset].item()))
                    if predictions[segments_offset]:
                        sequence.append(flatten_segments[segments_offset])
                    segments_offset += 1

                pred_sequences.append(self.sequence_delimiter.join(sequence))
                pred_classification.append(classification)

        return ExtSummarizationResultOutput(sample_ids=batch.sample_ids,
                                            logits=logits,
                                            summary=pred_sequences,
                                            classification=pred_classification)


class StructPreTrainedModel(PreTrainedModel):
    config_class = StructConfig
    base_model_prefix = "struct"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class NArySearch(torch.nn.Module):

    def __init__(self, config: StructConfig):
        super().__init__()
        self.config = config

        self.query_projection = torch.nn.Linear(config.n_ary_search_dim, config.n_ary_search_dim)
        self.key_projection = torch.nn.Linear(config.n_ary_search_dim, config.n_ary_search_dim)
        self.value_projection = torch.nn.Linear(config.n_ary_search_dim, config.n_ary_search_dim)
        self.dropout = torch.nn.Dropout(config.nary_search_dropout_prob)

    @staticmethod
    @torch.no_grad()
    def assemble_edge_index(parent_indices: torch.Tensor, tree_index: torch.Tensor, num_of_children: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assembles edge index from tree groups.

        :param parent_indices: index of the parent node for each query
            [all_tokens]
        :param tree_index: children for each parent
            parent children are represented by intervals [min, max) when the parent has no children the interval is [s,s] (e.g., [0,0])
            [all_tokens, 2]
        :param num_of_children: number of children for each parent
            [all_tokens]
        :return:
            edge index
                0 from query token
                1 remapped children indices so the indices start from 0
                2 to key token
                [3, num_edges]
            unique_children_indices
                is useful for selecting the right children embeddings
        """
        start_time = time.time()
        parents_subtree_index = tree_index[parent_indices]

        edge_index = torch.empty(3,
                                 num_of_children.sum(),
                                 dtype=torch.long,
                                 device=parent_indices.device)

        edge_index[0] = torch.repeat_interleave(torch.arange(parent_indices.size(0),
                                                             device=parent_indices.device,
                                                             dtype=torch.long),
                                                 num_of_children,
                                                 dim=0)

        edge_index[2] = mrange(parents_subtree_index)

        unique_children_indices = torch.unique(edge_index[2])

        indices_map = torch.zeros(tree_index.size(0), dtype=torch.long, device=tree_index.device)
        indices_map[unique_children_indices] = torch.arange(unique_children_indices.size(0), dtype=torch.long,
                                                            device=unique_children_indices.device)
        edge_index[1] = indices_map[edge_index[2]]

        return edge_index, unique_children_indices

    def forward(self, queries: torch.Tensor, tree_embeddings: torch.Tensor,
                tree_index: torch.Tensor, root_index: torch.Tensor) -> torch.Tensor:
        """
        Computes nary search representation.

        :param queries: queries for which we want to perform nary search
            [all_queries, n_ary_search_dim]
        :param tree_embeddings: embeddings of the tree nodes
            [all_nodes, n_ary_search_dim]
        :param tree_index: indices of children for each parent
            parent children are represented by intervals [min, max) when the parent has no children the interval is [s,s] (e.g., [0,0])
        :param root_index: index of the root node for each query
            [all_queries]
        :return: aggregated embeddings for each query
            [all_queries, n_ary_search_dim]
        """
        res = []
        parent_index = root_index

        queries = self.query_projection(queries)

        lvl = 0
        while True:
            cycle_start = time.time()

            parents_subtree_index = tree_index[parent_index]
            num_of_children = parents_subtree_index[:, 1] - parents_subtree_index[:, 0]

            if num_of_children.min() == 0:
                break

            edges, unique_children_indices = self.assemble_edge_index(parent_index, tree_index, num_of_children)
            # -> [3, num_edges]

            children_embeddings = tree_embeddings[unique_children_indices]

            key_embeddings = self.key_projection(children_embeddings)
            # -> [children_embeddings, pooling_size]

            scores = indices_dot_product(queries, key_embeddings, edges[:2])
            # -> [num_edges]

            probs = softmax(scores, edges[0], num_nodes=edges[0].size(0))

            if scores.dtype != probs.dtype:
                # the softmax is converting half to float we need to convert it back
                probs = probs.to(scores.dtype)

            probs = self.dropout(probs)

            val_embeddings = self.value_projection(children_embeddings)

            res.append(indices_weighted_scatter_sum(val_embeddings, edges[:2], probs))

            parent_index = edges[2][scatter_argmax(probs, edges[0], dim=0, dim_size=queries.size(0))]

            lvl += 1

        return torch.cat(res, dim=-1)


class StructGenerate(torch.nn.Module):
    def __init__(self, config: StructConfig, construction: HFStructConstructionModel):
        super().__init__()

        self.config = config
        self.construction = construction

        self.pooling_for_search = StructHierPooling(config, position_embeddings=None, vocab_input=False,
                                                    for_nary_search=True)    # also serves as transformation to get homogenous number of dimensions

        self.sources_heads = torch.nn.ModuleList([
            torch.nn.ModuleList([NArySearch(config=self.config) for _ in range(self.config.num_heads)])
            for _ in range(self.config.num_sources)
        ])

        init_states = [torch.nn.Parameter(torch.normal(0, self.config.init_std, (1, self.config.n_ary_search_dim)))]
        for max_lvl_context in reversed(self.config.levels):
            init_states.append(
                torch.nn.Parameter(torch.normal(0, self.config.init_std, (max_lvl_context, self.config.n_ary_search_dim)))
            )

        self.init_states = torch.nn.ParameterList(init_states)
        self.gen_transform = torch.nn.ModuleList([
            torch.nn.Linear(
                sum(self.config.embedding_size_for_level(i) for i in range(len(self.config.levels), i, -1))+(self.config.n_ary_search_dim*(1+len(self.config.levels)))*self.config.num_heads*self.config.num_sources,
                self.config.embedding_size_for_level(i)
            ) for i in reversed(range(len(self.config.levels)+1))
        ])

        self.query_projection = torch.nn.ModuleList([
            torch.nn.Linear(
                self.config.n_ary_search_dim + sum(self.config.embedding_size_for_level(lvl) for lvl in range(len(self.config.levels), i, -1)),
                self.config.n_ary_search_dim
            ) for i in reversed(range(len(self.config.levels)))
        ])

        self.homogenous = self.config.is_homogenous()

        self.expand_gen_space_cls = torch.nn.ModuleList([
            torch.nn.Linear(
                sum(self.config.embedding_size_for_level(lvl) for lvl in range(len(self.config.levels), i-1, -1)),
                max_lvl_context
            ) for i, max_lvl_context in zip(range(len(self.config.levels), 0, -1), reversed(self.config.levels))
        ])

        self.freeze_construction = False

    def construct_tree(self, inputs: torch.Tensor, batch_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Constructs tree from the input sequence.

        :param inputs: input sequence
            [all_tokens]
        :param batch_index: batch index
        :return:
            tree embeddings
            tree index
            root index
        """

        with torch.no_grad() if self.freeze_construction else nullcontext():
            multi_lvl_batch_index, original_batch_sample_identifier = self.construction.construction.layers[0].hier_pooling.prepare_multi_lvl_batch_index(batch_index)
            hidden_states = self.construction(inputs, batch_index, multi_lvl_batch_index=multi_lvl_batch_index)
        return self.pooling_for_search(hidden_states, multi_lvl_batch_index, flattened=False,
                                                  original_batch_sample_identifier=original_batch_sample_identifier)

    def search_in_tree(self, tree_embeddings: torch.Tensor, tree_index: torch.Tensor, queries: torch.Tensor,
                       queries_root_index: torch.Tensor, heads) -> torch.Tensor:
        """
        Searches in the tree.

        :param tree_embeddings: embeddings of the tree nodes
        :param tree_index: indices of children for each parent
        :param queries: queries
            [all_queries, n_ary_search_dim]
        :param queries_root_index: index of the root node for each query
        :param heads: nary search heads
        :return: aggregated embeddings for each query
            [queries, n_ary_search_dim]
        """

        res = []

        for head in heads:
            res.append(head(queries, tree_embeddings, tree_index, queries_root_index))

        return torch.cat(res, dim=0)

    def gen_space_expansion(self, gen_nodes: torch.Tensor, lvl_from_top: int, rightmost_node_index: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Provides nodes selection index for gen space expansion.

        :param gen_nodes: nodes for expansion
            [nodes, embedding_size_for_lvl(n)]
        :param lvl_from_top: level from top
        :param rightmost_node_index: index of the rightmost node for each sample
            [queries]
        :return:
            selection index
                [number of expanded nodes]
            expansion scores for each rightmost node
                [number of rightmost nodes, max_lvl_context]
        """

        classification_res = self.expand_gen_space_cls[lvl_from_top](gen_nodes[rightmost_node_index])

        selection_index = torch.arange(gen_nodes.shape[0], device=gen_nodes.device)
        repeat_n_times = torch.full((gen_nodes.shape[0],), len(self.init_states[lvl_from_top]), dtype=torch.long, device=gen_nodes.device)
        repeat_n_times[rightmost_node_index] = torch.argmax(classification_res, dim=-1) + 1

        return selection_index.repeat_interleave(repeat_n_times), classification_res

    def forward(self, inputs: List[torch.Tensor], batch_index: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        trees = [self.construct_tree(i, b) for i, b in zip(inputs, batch_index)]

        init_state_for_query = self.init_states[0].expand(trees[0][2].size(0), -1)  # one query for each sample
        search_res = torch.cat([
            self.search_in_tree(
                tree[0],
                tree[1],
                init_state_for_query,
                tree[2],
                self.sources_heads[i]
            ) for i, tree in enumerate(trees)
        ], dim=-1)
        # -> [samples, n_ary_search_dim*num_heads*num_sources]
        gen_nodes = self.gen_transform[0](search_res)
        # -> [samples, embedding_size_for_lvl(n)]

        expansion_cls_res = []

        gen_space_batch_index = torch.arange(gen_nodes.size(0), device=gen_nodes.device)

        for lvl_from_top, init_states_for_lvl in zip(range(0, len(self.gen_transform)), self.init_states[1:]):
            rightmost_node_index = torch.unique(gen_space_batch_index, return_counts=True)[1].cumsum(0) - 1
            index_selection, classification_res = self.gen_space_expansion(gen_nodes, lvl_from_top, rightmost_node_index)
            expansion_cls_res.append(classification_res)

            expanded_gen_nodes = gen_nodes[index_selection]
            gen_space_batch_index = gen_space_batch_index[index_selection]

            queries_src = torch.zeros(
                expanded_gen_nodes.size(0),
                gen_nodes.shape[1] + self.config.n_ary_search_dim,
                device=gen_nodes.device,
                dtype=gen_nodes.dtype
            )
            queries_src[:, :gen_nodes.shape[1]] = expanded_gen_nodes

            index_selection = torch.arange(expanded_gen_nodes.size(0), device=expanded_gen_nodes.device) % init_states_for_lvl.size(0)
            queries_src[:, gen_nodes.shape[1]:] = init_states_for_lvl[index_selection]

            queries = self.query_projection[lvl_from_top](queries_src)

            search_res = torch.cat([
                self.search_in_tree(
                    tree[0],
                    tree[1],
                    queries,
                    tree[2][gen_space_batch_index],
                    self.sources_heads[i]
                ) for i, tree in enumerate(trees)
            ], dim=-1)
            # -> [nodes, num_of_lvls*n_ary_search_dim*num_heads*num_sources]

            gen_nodes_src = torch.zeros(
                queries_src.shape[0],
                gen_nodes.shape[1]+search_res.shape[1],
                device=gen_nodes.device,
                dtype=gen_nodes.dtype
            )

            gen_nodes_src[:, :gen_nodes.shape[1]] = expanded_gen_nodes
            gen_nodes_src[:, gen_nodes.shape[1]:] = search_res

            gen_nodes = torch.zeros(
                queries_src.shape[0],
                gen_nodes.shape[1] + self.gen_transform[lvl_from_top+1].out_features,
                device=gen_nodes.device,
                dtype=gen_nodes.dtype
            )
            gen_nodes[:, :expanded_gen_nodes.shape[1]] = expanded_gen_nodes
            gen_nodes[:, expanded_gen_nodes.shape[1]:] = self.gen_transform[lvl_from_top+1](gen_nodes_src)
            # -> [nodes, sum(embedding_size_for_lvl(n) to embedding_size_for_lvl(i))]

            print(f"Level {lvl_from_top}", gen_nodes.shape, gen_space_batch_index.shape, expansion_cls_res[-1].shape,
                  torch.all(torch.isfinite(gen_nodes)).item(), torch.all(torch.isfinite(gen_space_batch_index)).item(), torch.all(torch.isfinite(expansion_cls_res[-1])).item())

        return gen_nodes, gen_space_batch_index, expansion_cls_res


class HFStruct(StructPreTrainedModel):

    def __init__(self, config: StructConfig):
        super().__init__(config)
        self.construction = HFStructConstructionModel(config)
        self.generator = StructGenerate(config, self.construction)

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.construction.vocabulary_embeddings.embeddings

    def set_input_embeddings(self, value: torch.nn.Module):
        self.construction.vocabulary_embeddings.embeddings = value

    def forward(self, inputs: List[torch.Tensor], batch_index: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        return self.generator(inputs, batch_index)


class HFStructForTokenClassification(StructPreTrainedModel):
    def __init__(self, config: StructConfig):
        super().__init__(config)

        self.model = HFStruct(config)

        self.lm_head = torch.nn.Linear(config.hier_embedding_size, config.num_labels, bias=False)
        self.config = config
        self.post_init()

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: torch.nn.Module):
        self.model.set_input_embeddings(value)

    def forward(self, inputs: List[torch.Tensor], batch_index: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Computes output of the model.

        :param inputs: Input ids.
        :param batch_index: Batch index.
        :return:
            hierarchical embeddings of the output sequence
            batch index for the output embeddings
            prediction scores
            expansion classification results
        """

        outputs, outputs_batch_index, exp_cls_res = self.model(inputs, batch_index)

        with torch.no_grad() if self.model.generator.freeze_construction else nullcontext():
            prediction_scores = self.lm_head(outputs)

        return outputs, outputs_batch_index, prediction_scores, exp_cls_res


class HFStructWithLMHead(HFStructForTokenClassification):
    """
    Same as HFStructForTokenClassification but it will set the vocab_size as num_labels.
    """

    def __init__(self, config: StructConfig):

        config.num_labels = config.vocab_size
        super().__init__(config)

# The MIT License (MIT)

# Copyright (C) 2021-2023 ExplosionAI GmbH

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from dataclasses import dataclass
from typing import Optional

import torch

from ..layers.activations import Activation


@dataclass
class RotaryEmbeddingConfig:
    """
    Configuration options for rotary embeddings (`Su et al., 2021`_).

    .. _Su et al., 2021: https://arxiv.org/abs/2104.09864

    :param rotary_base:
        Base in signifying the rotary embedding period.
    :param rotary_fraction:
        Fraction of hidden width to apply rotary embeddings to.
        Must be in ``[0,1]``.
    """

    rotary_base: int
    rotary_fraction: float


@dataclass
class TransformerAttentionLayerConfig:
    """
    Configuration options for self-attention.

    :param dropout_prob:
        Dropout probabilty to apply after attention.
    :param hidden_width:
        Hidden width of the transformer.
    :param n_query_heads:
        Number of attention heads.
    :param n_key_value_heads:
        Number of key and value heads.
    :param rotary_embeddings:
        Rotary embedding configuration.
    :param use_alibi:
        Use ALiBi linear biases.
    :param use_bias:
        Use bias in linear layers.
    :param use_parallel_attention:
        Use parallel attention.
    :param output_attention_probs:
        Output attention probabilities (i.e. before matmul with value).
    """

    dropout_prob: float
    hidden_width: int
    n_query_heads: int
    n_key_value_heads: int
    rotary_embeddings: Optional[RotaryEmbeddingConfig]
    use_alibi: bool
    use_bias: bool
    use_parallel_attention: bool
    output_attention_probs: bool

@dataclass
class TransformerEmbeddingLayerConfig:
    """
    Configuration options for embeddings.

    :param dropout_prob:
        Dropout probabilty for the embedding layer.
    :param embedding_width:
        Width of the embedding representations.
    :param layer_norm_eps:
        Epsilon for layer normalization.
    :param n_positions:
        Maximum length of position embeddings.
    :param n_pieces:
        Vocabulary size (number of embeddings).
    :param n_types:
        Token type vocabulary size (number of token type embeddings).
    """

    dropout_prob: float
    embedding_width: int
    layer_norm_eps: float
    n_positions: Optional[int]
    n_pieces: int
    n_types: Optional[int]


@dataclass
class TransformerFeedForwardLayerConfig:
    """
    Configuration options for transformer feed-forward layers.

    :param activation:
        Activation in the feed-forward layer
    :param hidden_width:
        Hidden width of the transformer.
    :param intermediate_width:
        Intermediate width in the feed-forward layer.
    :param use_bias:
        Use bias in linear layers.
    :param use_gate:
        Use Gated Linear Units.
    """

    activation: Activation
    hidden_width: int
    intermediate_width: int
    use_bias: bool
    use_gate: bool


@dataclass
class TransformerLayerConfig:
    """
    Configuration options for transformer layers.

    :param attention:
        Attention layer config.
    :param dropout_prob:
        Dropout probabilty to apply after hidden layers.
    :param feedforward:
        Feed-forward layer config.
    :param layer_norm_eps:
        Epsilon for layer normalization.
    :param n_hidden_layers:
        Number of hidden layers.
    """

    attention: TransformerAttentionLayerConfig
    dropout_prob: float
    feedforward: TransformerFeedForwardLayerConfig
    layer_norm_eps: float
    n_hidden_layers: int


@dataclass
class TransformerConfig:
    """
    Configuration options for a transformer model.

    :param embedding:
        Embedding layer config.
    :param layer:
        Transformer hidden layer config.
    :param dtype:
        Default data type used by the model's
        parameters.
    """

    embedding: TransformerEmbeddingLayerConfig
    layer: TransformerLayerConfig
    dtype: torch.dtype
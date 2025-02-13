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

from abc import abstractmethod
from typing import Generic, List, Optional, TypeVar

from torch import Tensor
from torch.nn import Module

from ..layers.attention import AttentionMask
from .config import TransformerConfig
from .output import CacheT, CausalLMOutputWithCache, ModelOutput, ModelOutputWithCache

ConfigT = TypeVar("ConfigT", bound=TransformerConfig)


class TransformerModule(Generic[ConfigT], Module):
    """
    Base class for transformer modules.
    """

    _config: ConfigT

    def __init__(self, config: ConfigT):
        super().__init__()

        self._config = config

    @property
    def config(self) -> ConfigT:
        """
        Returns the model's configuration.
        """
        return self._config


class CausalLMModule(Generic[ConfigT, CacheT], TransformerModule[ConfigT]):
    """
    Base class for causal language model modules.
    """

    def __init__(self, config: ConfigT):
        super().__init__(config)

    @abstractmethod
    def forward(
        self,
        piece_ids: Tensor,
        attention_mask: AttentionMask,
        *,
        cache: Optional[List[CacheT]] = None,
        positions: Optional[Tensor] = None,
        store_cache: bool = False,
    ) -> CausalLMOutputWithCache[CacheT]:
        """
        Apply the causal language model to the given piece identifiers.

        :param piece_ids:
            Piece identifiers to apply the decoder to.

            *Shape:* ``(batch_size, seq_len)``
        :param attention_mask:
            Attention mask. Sequence elements for which the
            corresponding mask element is set to ``False`` are ignored
            during attention calculation.
        :param cache:
            Key/value cache to avoid recomputing key/value representations
            for tokens that were previously seen.
        :param positions:
            Input positions. Positions are needed to look up rotary embeddings.
            Normally, these positions are calculated automatically. But if the
            positions deviate for some reason, they can be provided through this argument.

            *Shape:* ``(batch_size, seq_len)``
        :param store_cache:
            Whether to cache the key/value representations for future reuse.
        :returns:
            Causal language model output with key/value cache.
        """
        raise NotImplementedError


class DecoderModule(Generic[ConfigT, CacheT], TransformerModule[ConfigT]):
    """
    Base class for decoder modules.
    """

    def __init__(self, config: ConfigT):
        super().__init__(config)

    @abstractmethod
    def forward(
        self,
        piece_ids: Tensor,
        attention_mask: AttentionMask,
        *,
        cache: Optional[List[CacheT]] = None,
        positions: Optional[Tensor] = None,
        store_cache: bool = False,
    ) -> ModelOutputWithCache[CacheT]:
        """
        Apply the decoder to the given piece identifiers.

        :param piece_ids:
            Piece identifiers to apply the decoder to.

            *Shape:* ``(batch_size, seq_len)``
        :param attention_mask:
            Attention mask. Sequence elements for which the
            corresponding mask element is set to ``False`` are ignored
            during attention calculation.
        :param cache:
            Key/value cache to avoid recomputing key/value representations
            for tokens that were previously seen.
        :param positions:
            Input positions. Positions are needed to look up position embeddings.
            Normally, these positions are calculated automatically. But if the
            positions deviate for some reason, they can be provided through this argument.

            *Shape:* ``(batch_size, seq_len)``
        :param store_cache:
            Whether to cache the key/value representations for future reuse.
        :returns:
            Decoder output with key/value cache.
        """
        raise NotImplementedError


class EncoderModule(Generic[ConfigT], TransformerModule[ConfigT]):
    """
    Base class for encoder modules.
    """

    def __init__(self, config: ConfigT):
        super().__init__(config)

    @abstractmethod
    def forward(
        self,
        piece_ids: Tensor,
        attention_mask: AttentionMask,
        *,
        positions: Optional[Tensor] = None,
        type_ids: Optional[Tensor] = None,
    ) -> ModelOutput:
        """
        Apply the encoder to the input.

        :param piece_ids:
            Piece identifiers to apply the encoder to.

            *Shape:* ``(batch_size, seq_len)``
        :param attention_mask:
            Attention mask. Sequence elements for which the
            corresponding mask element is set to ``False`` are ignored
            during attention calculation.
        :param positions:
            Input positions. Positions are used to look up position embeddings.
            Normally, these positions are calculated automatically. But if the
            positions deviate for some reason, they can be provided through this argument.

            *Shape:* ``(batch_size, seq_len)``
        :param type_ids:
            Type identifiers to indicate the spans of different
            sequences in the input. Useful when performing tasks like
            sequence classification and question answering.

            *Shape:* ``(batch_size, seq_len)``
        :returns:
            Encoder output.
        """
        raise NotImplementedError
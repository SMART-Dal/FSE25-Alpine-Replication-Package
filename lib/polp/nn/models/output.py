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
from typing import Generic, List, Optional, TypeVar, Union

from torch import Tensor

from ..layers.cache import CacheProtocol
from ..util.dataclass import DataclassAsTuple

CacheT = TypeVar("CacheT", bound=CacheProtocol)


@dataclass
class ModelOutput(DataclassAsTuple):
    """
    Base class for model outputs.

    :param all_outputs:
        The first element is the output of the embedding layer. The
        rest of the elements are the states of each encoder hidden
        layer respectively.
    :param attention_probs:
        Attention probabilities of each head at each layer.

        *Shape:* ``(n_hidden_layers, n_heads, seq_len, seq_len)``
    """

    all_outputs: List[Tensor]
    attention_probs: Optional[Tensor]

    @property
    def embedding_layer(self) -> Tensor:
        """
        Return the output of the embedding layer.

        :returns:
            Embedding layer output.

            *Shape:* ``(batch_size, seq_len, width)``
        """
        return self.all_outputs[0]

    def hidden_layer_states(self, idx: int) -> Tensor:
        """
        Return the hidden representations of a given layer.

        :param idx:
            Layer index. Must be in ``[0, n_hidden_layers)``.
        :returns:
            Hidden representation of the layer.

            *Shape:* ``(batch_size, seq_len, width)``
        """
        if 0 <= idx < len(self.all_outputs) - 1:
            return self.all_outputs[idx + 1]
        else:
            raise ValueError(
                "Attempting to select a transformer output tensor using an invalid "
                f"layer index ({idx}). Expected range: 0 <= idx < {(len(self.all_outputs) - 1)}"
            )

    @property
    def last_hidden_layer_state(self) -> Tensor:
        """
        Return the hidden representation of the last layer.

        :returns:
            Last hidden representation of the last layer.

            *Shape:* ``(batch_size, seq_len, width)``
        """
        return self.all_outputs[-1]
    
    @property
    def attention_probabilities(self) -> Union[Tensor, None]:
        """
        Return the attention probabilities of all the heads.

        :returns:   
            Attention probabilities of all the heads across all the layers.
            
            *Shape:* ``(n_hidden_layers, n_heads, seq_len, seq_len)``
        """
        return self.attention_probs

    @property
    def all_hidden_layer_states(self) -> List[Tensor]:
        """
        Return the hidden representation of all the layers.

        :returns:
            Hidden representations of all the layers.

            *Shape:* ``(batch_size, seq_len, width)``
        """
        return self.all_outputs[1:]


@dataclass
class ModelOutputWithCache(Generic[CacheT], ModelOutput):
    """
    Output of decoder modules.

    :param cache:
        Model cache. The cache can be used with future calls
        to a model to reuse computations for efficiency
    """

    cache: Optional[List[CacheT]]


@dataclass
class CausalLMOutputWithCache(Generic[CacheT], ModelOutputWithCache[CacheT]):
    """
    Output of causal language model modules.

    :param logits:
        Logits of the distributions of predicted tokens.
    """

    logits: Tensor
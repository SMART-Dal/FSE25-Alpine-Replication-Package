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

from typing import Generic, List, Optional

import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from ..layers.attention import AttentionMask
from ..layers.cache import KeyValueCache
from .module import CausalLMModule, ConfigT, DecoderModule, EncoderModule
from .output import CausalLMOutputWithCache, ModelOutput, ModelOutputWithCache


class TransformerDecoder(Generic[ConfigT], DecoderModule[ConfigT, KeyValueCache]):
    """
    Transformer decoder (`Vaswani et al., 2017`_) base class.

    This class provides an implementation of the ``forward`` method.
    Subclasses must set the given member attributes.

    .. _Vaswani et al., 2017: https://arxiv.org/abs/1706.03762
    """

    embeddings: Module
    layers: ModuleList
    output_layer_norm: Module

    def forward(
        self,
        piece_ids: Tensor,
        attention_mask: AttentionMask,
        *,
        cache: Optional[List[KeyValueCache]] = None,
        positions: Optional[Tensor] = None,
        store_cache: bool = False,
    ) -> ModelOutputWithCache[KeyValueCache]:
        embeddings = self.embeddings(piece_ids)
        layer_output = embeddings

        layer_outputs = []
        new_cache = []
        layer_cache = None
        for layer in self.layers:
            if cache is not None:
                # The key-value cache is stored per layer, so peel off one
                # layer at a time.
                layer_cache = cache[0]
                cache = cache[1:]
            layer_output, new_layer_cache = layer(
                layer_output,
                attention_mask,
                cache=layer_cache,
                store_cache=store_cache,
                positions=positions,
            )
            layer_outputs.append(layer_output)
            if store_cache:
                new_cache.append(new_layer_cache)

        layer_outputs[-1] = self.output_layer_norm(layer_outputs[-1])

        return ModelOutputWithCache(
            all_outputs=[embeddings, *layer_outputs],
            cache=new_cache if store_cache else None,
        )


class TransformerCausalLM(Generic[ConfigT], CausalLMModule[ConfigT, KeyValueCache]):
    """
    Transformer causal LM (`Vaswani et al., 2017`_) base class.

    This class provides an implementation of the ``forward`` method.
    Subclasses must set the given member attributes..

    .. _Vaswani et al., 2017: https://arxiv.org/abs/1706.03762
    """

    decoder: TransformerDecoder
    output_embeddings: Module

    def forward(
        self,
        piece_ids: Tensor,
        attention_mask: AttentionMask,
        cache: Optional[List[KeyValueCache]] = None,
        positions: Optional[Tensor] = None,
        store_cache: bool = False,
    ) -> CausalLMOutputWithCache[KeyValueCache]:
        decoder_output = self.decoder(
            piece_ids,
            attention_mask,
            cache=cache,
            store_cache=store_cache,
            positions=positions,
        )
        logits = self.output_embeddings(decoder_output.last_hidden_layer_state)
        return CausalLMOutputWithCache(
            all_outputs=decoder_output.all_outputs,
            cache=decoder_output.cache,
            logits=logits,
        )


class TransformerEncoder(Generic[ConfigT], EncoderModule[ConfigT]):
    """
    Transformer encoder (`Vaswani et al., 2017`_) base class.

    This class provides an implementation of the ``forward`` method.
    Subclasses must set the given member attributes.

    .. _Vaswani et al., 2017: https://arxiv.org/abs/1706.03762
    """

    embeddings: Module
    layers: ModuleList

    def forward(
        self,
        piece_ids: Tensor,
        attention_mask: AttentionMask,
        *,
        positions: Optional[Tensor] = None,
        type_ids: Optional[Tensor] = None,
    ) -> ModelOutput:
        embeddings = self.embeddings(piece_ids, positions=positions, type_ids=type_ids)
        layer_output = embeddings

        layer_outputs = []
        attention_probes = []
        for layer in self.layers:
            if self.config.layer.attention.output_attention_probs:
                layer_output, layer_attention_probes, _ = layer(layer_output, attention_mask)
                attention_probes.append(layer_attention_probes)
            else:
                layer_output, _ = layer(layer_output, attention_mask)
            layer_outputs.append(layer_output)

        return ModelOutput(all_outputs=[embeddings, *layer_outputs], attention_probs=attention_probes)
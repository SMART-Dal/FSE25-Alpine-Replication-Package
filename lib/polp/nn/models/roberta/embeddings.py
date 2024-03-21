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

from typing import Optional

import torch
from torch import Tensor

from ...layers.transformer import (
    EmbeddingDropouts,
    EmbeddingLayerNorms,
    TransformerEmbeddings,
)


class RoBERTaEmbeddings(TransformerEmbeddings):
    """
    RoBERTa (`Liu et al., 2019`_) embedding layer.

    This layer only differs from
    :py:class:`~curated_transformers.layers.transformer.TransformerEmbeddingsLayer`
    in the computation of positions.

    .. _Liu et al., 2019: https://arxiv.org/abs/1907.11692
    """

    def __init__(
        self,
        *,
        dropouts: EmbeddingDropouts,
        embedding_width: int,
        hidden_width: int,
        layer_norms: EmbeddingLayerNorms,
        n_pieces: int,
        n_positions: Optional[int],
        n_types: Optional[int],
        padding_id: int,
        device: Optional[torch.device] = None,
    ):
        """
        Construct a RoBERTa embeddings layer.

        :param dropouts:
            Dropouts to use in the embeddings layer.
        :param embedding_width:
            Width of the embeddings.
        :param hidden_width:
            Hidden width of the transformer. If this width differs from
            ``embedding_width``, a projection layer is added to ensure
            that the output of the embeddings layer has the same width as the
            transformer.
        :param layer_norms:
            Layer norms to use in the embeddings layer.
        :param n_pieces:
            Number of piece embeddings.
        :param n_positions:
            Number of position embeddings. Position embeddings are disabled
            by using ``None``. Position embeddings can be used to inform the
            model of input order.
        :param n_types:
            Number of type embeddings. Type embeddings are disabled by using
            ``None``. Type embeddings can be used to inform the model of the
            spans of different sequences in the input.
        :param padding_id:
            Padding identifier. The padding identifier is used to compute the
            positions when they are not provided to the ``forward`` method.
        :param device:
            Device on which the module is to be initialized.
        """

        super().__init__(
            dropouts=dropouts,
            embedding_width=embedding_width,
            hidden_width=hidden_width,
            layer_norms=layer_norms,
            n_pieces=n_pieces,
            n_positions=n_positions,
            n_types=n_types,
            device=device,
        )

        self.padding_id = padding_id

    def _get_positions(self, x: Tensor) -> Tensor:
        # We need to generate the position IDs from the
        # input tensor to pass to the embedding layer and
        # handle padding, c.f https://github.com/huggingface/transformers/blob/330247ede2d8265aae9ab0b7a0d1a811c344960d/src/transformers/models/roberta/modeling_roberta.py#L1566

        mask = x.ne(self.padding_id).int()
        return (mask.cumsum(dim=1) * mask) + self.padding_id

    def forward(
        self,
        piece_ids: Tensor,
        *,
        positions: Optional[Tensor] = None,
        type_ids: Optional[Tensor] = None,
    ) -> Tensor:
        if positions is None:
            positions = self._get_positions(piece_ids)
        return super().forward(
            piece_ids,
            positions=positions,
            type_ids=type_ids,
        )
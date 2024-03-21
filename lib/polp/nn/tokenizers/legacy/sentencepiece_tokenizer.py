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

from typing import Iterable, List, Optional

from curated_tokenizers import SentencePieceProcessor

from ..chunks import MergedInputChunks, MergedSpecialPieceChunk
from ..tokenizer import PiecesWithIds
from .legacy_tokenizer import LegacyTokenizer


class SentencePieceTokenizer(LegacyTokenizer):
    """
    Piece tokenizer using SentencePiece encoding (`Kudo et al., 2018`_).

    .. _Kudo et al., 2018: https://arxiv.org/abs/1808.06226
    """

    def __init__(
        self,
        *,
        processor: SentencePieceProcessor,
    ):
        """
        Construct a tokenizer from ``curated-tokenizers`` SentencePiece processor.

        :param processor:
            The processor to wrap.
        """
        self.processor = processor
        self._eos_piece = processor.id_to_piece(processor.eos_id())

    @property
    def eos_piece(self) -> Optional[str]:
        return self._eos_piece

    def piece_to_id(self, piece: str) -> Optional[int]:
        return self.processor.piece_to_id(piece)

    def _decode(
        self, input: Iterable[Iterable[int]], skip_special_pieces: bool
    ) -> List[str]:
        # skip_special_pieces is currently ignored. Since sentencepiece
        # processes the whole input output, this probably needs to be
        # handled by the sentencepiece library itself?
        return [self.processor.decode_from_ids(ids) for ids in input]

    def _encode(self, input: Iterable[MergedInputChunks]) -> PiecesWithIds:
        ids = []
        pieces = []

        for seq in input:
            seq_ids = []
            seq_pieces = []

            for chunk in seq:
                if isinstance(chunk, MergedSpecialPieceChunk):
                    piece_id = self.processor.piece_to_id(chunk.piece)
                    if piece_id is None:
                        raise ValueError(f"Unknown special piece: {chunk.piece}")
                    seq_ids.append(piece_id)
                    seq_pieces.append(chunk.piece)
                else:
                    chunk_ids, chunk_pieces = self.processor.encode(chunk.text)
                    seq_ids.extend(chunk_ids)
                    seq_pieces.extend(chunk_pieces)

            ids.append(seq_ids)
            pieces.append(seq_pieces)

        return PiecesWithIds(ids=ids, pieces=pieces)
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

from typing import Dict, Iterable, List, Optional, Tuple, TypeVar

from curated_tokenizers import ByteBPEProcessor

from ..chunks import MergedInputChunks, MergedSpecialPieceChunk
from ..tokenizer import PiecesWithIds
from .legacy_tokenizer import LegacyTokenizer

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="ByteBPETokenizer")


class ByteBPETokenizer(LegacyTokenizer):
    """
    Piece tokenizer using byte-level byte pair encoding
    (`Gage, 1994`_, `Sennrich et al., 2016`_).

    .. _Gage, 1994: https://dl.acm.org/doi/10.5555/177910.177914
    .. _Sennrich et al., 2016: https://arxiv.org/abs/1508.07909
    """

    def __init__(
        self,
        *,
        vocab: Dict[str, int],
        merges: List[Tuple[str, str]],
        special_pieces: Optional[Dict[str, int]] = None,
    ):
        """
        Construct a byte BPE tokenizer.

        :param vocab:
            The word piece vocabulary.
        :param merges:
            Merges.
        :param special_pieces:
            Special pieces.
        """
        self.special_piece_to_id = {} if special_pieces is None else special_pieces
        self.id_to_special_piece = {v: k for k, v in self.special_piece_to_id.items()}
        vocab.update(self.special_piece_to_id)
        self.processor = ByteBPEProcessor(vocab, merges)

    def piece_to_id(self, piece: str) -> Optional[int]:
        return self.processor.token_to_id(piece)

    def _decode(
        self, input: Iterable[Iterable[int]], skip_special_pieces: bool
    ) -> List[str]:
        input = [
            [
                id
                for id in ids
                if not skip_special_pieces or id not in self.id_to_special_piece
            ]
            for ids in input
        ]
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
                    for idx, token in enumerate(chunk.text.split(" ")):
                        if idx != 0:
                            token = " " + token
                        token_ids, token_pieces = self.processor.encode(token)
                        seq_ids.extend(token_ids)
                        seq_pieces.extend(token_pieces)

            ids.append(seq_ids)
            pieces.append(seq_pieces)

        return PiecesWithIds(ids=ids, pieces=pieces)
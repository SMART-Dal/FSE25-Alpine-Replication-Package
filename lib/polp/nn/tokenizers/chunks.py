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

import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Union

ChunkT = Union["SpecialPieceChunk", "TextChunk"]
MergedChunkT = Union["MergedSpecialPieceChunk", "TextChunk"]


@dataclass
class MergedSpecialPieceChunk:
    """
    A chunk that contains a special piece. This piece is not tokenized, but
    looked up directly in the vocabulary.

    :param piece:
        Piece to look up in the vocabulary.
    """

    piece: str


@dataclass
class SpecialPieceChunk:
    """
    A chunk that contains a special piece. This piece is not tokenized, but
    looked up directly in the vocabulary. Can additionally store strings that
    should be appended to a text chunk before or prepended to a text chunk
    after the special piece.

    :param piece:
        Piece to look up in the vocabulary.
    :param after:
        Text to prepend to the succeeding text chunk.
    :param before:
        Text to append to the preceding text chunk.
    """

    piece: str
    after: Optional[str] = None
    before: Optional[str] = None


@dataclass
class TextChunk:
    """
    A chunk of text that should be tokenized.

    :param text:
        Text that should be tokenized.
    """

    text: str


class MergedInputChunks(List[MergedChunkT]):
    """
    A list of chunks in which consecutive text chunks and before/after
    texts of special piece chunks are merged.
    """

    pass


class InputChunks(List[ChunkT]):
    """
    A list of chunks.
    """

    def merge_text_chunks(self) -> MergedInputChunks:
        """
        Merge multiple contiguous text chunks and before/after text
        in special piece chunks.
        """
        new_chunks = MergedInputChunks()
        for chunk in self:
            last_is_text = new_chunks and isinstance(new_chunks[-1], TextChunk)
            if isinstance(chunk, TextChunk):
                if last_is_text:
                    new_chunks[-1].text += chunk.text  # type: ignore[union-attr]
                else:
                    new_chunks.append(dataclasses.replace(chunk))
            elif isinstance(chunk, SpecialPieceChunk):
                if chunk.before:
                    if last_is_text:
                        new_chunks[-1].text += chunk.before  # type: ignore[union-attr]
                    else:
                        new_chunks.append(TextChunk(chunk.before))
                new_chunks.append(MergedSpecialPieceChunk(chunk.piece))
                if chunk.after:
                    new_chunks.append(TextChunk(chunk.after))
            else:
                raise ValueError(f"Unknown chunk type: {type(chunk).__name__}")

        return new_chunks
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

from abc import ABC, abstractmethod
from typing import IO, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .repository import Repository


class TransactionContext(ABC):
    """
    A context manager that represents an active transaction in
    a repository.
    """

    @abstractmethod
    def open(self, path: str, mode: str, encoding: Optional[str] = None) -> IO:
        """
        Opens a file as a part of a transaction. Changes to the
        file are deferred until the transaction has completed
        successfully.

        :param path:
            The path to the file on the parent repository.
        :param mode:
            Mode to open the file with (see Python ``open``).
        :param encoding:
            Encoding to use when the file is opened as text.
        :returns:
            An I/O stream.
        :raises FileNotFoundError:
            When the file cannot be found.
        :raises OSError:
            When the file cannot be opened.
        """
        ...

    @property
    @abstractmethod
    def repo(self) -> "Repository":
        """
        :returns:
            The parent repository on which this transaction is performed.
        """
        ...

    @abstractmethod
    def __enter__(self):
        """
        Invoked when the context manager is entered.
        """
        ...

    @abstractmethod
    def __exit__(self, exc_type, exc_value, exc_tb):
        """
        Invoked when the context manager exits.
        """
        ...
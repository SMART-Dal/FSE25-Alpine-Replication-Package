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
from typing import Any, Dict, Mapping, Optional, Type, TypeVar

from fsspec import AbstractFileSystem
from huggingface_hub.utils import EntryNotFoundError

from ..repository.file import RepositoryFile
from ..repository.fsspec import FsspecArgs, FsspecRepository
from ..repository.hf_hub import HfHubRepository
from ..repository.repository import Repository, TokenizerRepository

SelfFromHFHub = TypeVar("SelfFromHFHub", bound="FromHFHub")


class FromHFHub(ABC):
    """
    Mixin class for downloading tokenizers from Hugging Face Hub.

    It directly queries the Hugging Face Hub to load the tokenizer from
    its configuration file.
    """

    @classmethod
    @abstractmethod
    def from_hf_hub_to_cache(
        cls: Type[SelfFromHFHub],
        *,
        name: str,
        revision: str = "main",
    ):
        """
        Download the tokenizer's serialized model, configuration and vocab files
        from Hugging Face Hub into the local Hugging Face cache directory.
        Subsequent loading of the tokenizer will read the files from disk. If the
        files are already cached, this is a no-op.

        :param name:
            Model name.
        :param revision:
            Model revision.
        """
        raise NotImplementedError

    @classmethod
    def from_fsspec(
        cls: Type[SelfFromHFHub],
        *,
        fs: AbstractFileSystem,
        model_path: str,
        fsspec_args: Optional[FsspecArgs] = None,
    ) -> SelfFromHFHub:
        """
        Construct a tokenizer and load its parameters from an fsspec filesystem.

        :param fs:
            Filesystem.
        :param model_path:
            The model path.
        :param fsspec_args:
            Implementation-specific keyword arguments to pass to fsspec
            filesystem operations.
        :returns:
            The tokenizer.
        """
        return cls.from_repo(
            repo=FsspecRepository(fs, model_path, fsspec_args),
        )

    @classmethod
    def from_hf_hub(
        cls: Type[SelfFromHFHub], *, name: str, revision: str = "main"
    ) -> SelfFromHFHub:
        """
        Construct a tokenizer and load its parameters from Hugging Face Hub.

        :param name:
            Model name.
        :param revision:
            Model revision.
        :returns:
            The tokenizer.
        """
        return cls.from_repo(
            repo=HfHubRepository(name=name, revision=revision),
        )

    @classmethod
    @abstractmethod
    def from_repo(
        cls: Type[SelfFromHFHub],
        repo: Repository,
    ) -> SelfFromHFHub:
        """
        Construct and load a tokenizer from a repository.

        :param repository:
            The repository to load from.
        :returns:
            Loaded tokenizer.
        """
        ...


SelfLegacyFromHFHub = TypeVar("SelfLegacyFromHFHub", bound="LegacyFromHFHub")


class LegacyFromHFHub(FromHFHub):
    """
    Subclass of :class:`.FromHFHub` for legacy tokenizers. This subclass
    implements the ``from_hf_hub`` method and provides through the abstract
    ``_load_from_vocab_files`` method:

    * The vocabulary files requested by a tokenizer through the
    ``vocab_files`` member variable.
    * The tokenizer configuration (when available).
    """

    vocab_files: Dict[str, str] = {}

    @classmethod
    @abstractmethod
    def _load_from_vocab_files(
        cls: Type[SelfLegacyFromHFHub],
        *,
        vocab_files: Mapping[str, RepositoryFile],
        tokenizer_config: Optional[Dict[str, Any]],
    ) -> SelfLegacyFromHFHub:
        """
        Construct a tokenizer from its vocabulary files and optional
        configuration.

        :param vocab_files:
            The resolved vocabulary files (in a local cache).
        :param tokenizer_config:
            The tokenizer configuration (when available).
        :returns:
            The tokenizer.
        """
        raise NotImplementedError

    @classmethod
    def from_hf_hub_to_cache(
        cls: Type[SelfLegacyFromHFHub],
        *,
        name: str,
        revision: str = "main",
    ):
        repo = TokenizerRepository(HfHubRepository(name, revision=revision))
        for _, filename in cls.vocab_files.items():
            _ = repo.file(filename)

        try:
            _ = repo.tokenizer_config()
        except EntryNotFoundError:
            pass

    @classmethod
    def from_repo(
        cls: Type[SelfLegacyFromHFHub],
        repo: Repository,
    ) -> SelfLegacyFromHFHub:
        repo = TokenizerRepository(repo)
        vocab_files = {}
        for vocab_file, filename in cls.vocab_files.items():
            vocab_files[vocab_file] = repo.file(filename)

        try:
            tokenizer_config = repo.tokenizer_config()
        except OSError:
            tokenizer_config = None

        return cls._load_from_vocab_files(
            vocab_files=vocab_files, tokenizer_config=tokenizer_config
        )
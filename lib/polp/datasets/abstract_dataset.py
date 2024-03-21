from abc import ABC, abstractmethod
from typing import Callable, Optional

class AbstractDataset(ABC):

    def __init__(self, root: str, transform: Optional[Callable] = None) -> None:
        """Initiate abstract dataset.

        Parameters
        ----------
        root : str
            _description_
        transform : Optional[Callable], optional
            _description_, by default None
        """
        super().__init__()
        self.root = root
        self.transform = transform

    @abstractmethod
    def download(self) -> str:
        """Download the raw file(s) of a given dataset.

        Returns
        -------
        str
            _description_
        """
        raise NotImplementedError

    def init_dataloaders(self) -> None:
        """Initialize the dataloaders of the dataset.

        Returns
        -------
        None
            _description_
        """
        raise NotImplementedError
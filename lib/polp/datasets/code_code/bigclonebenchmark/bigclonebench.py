from typing import Callable, Optional, List, Tuple, Dict, Union, Any
import random, math

from polp.datasets import AbstractDataset, DatapointDataset, download_url
from polp.utils.io import read_file, read_jsonl
from . import BCBDatapoint



import os

class BigCloneBench(AbstractDataset):

    url = "https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/Code-Code/Clone-detection-BigCloneBench/dataset/"

    def __init__(
            self,
            root: str,
            default_train: Optional[bool] = True,
            split: Optional[Tuple[int]] = (80, 10, 10),
            transform: Optional[Callable[[BCBDatapoint], BCBDatapoint]] = None,) -> None:
        """Initiate BigCloneBench dataset filtered according to the `"Detecting Code Clones with 
        Graph Neural Network and Flow-Augmented Abstract Syntax Tree" <https://arxiv.org/abs/2002.08653>`_ paper.

        Parameters
        ----------
        root : str
            Directory where the dataset will be saved
        default_train : Optional[bool], optional
            If True, default train split is used, otherwise the train/test/val splits are randomly created, by default True
        split : Optional[Tuple[int]], optional
            Tuple of integers representing the percentage of the dataset to be used for the train, test and validation splits, respectively, by default [80, 10, 10]]
        transform : Optional[Callable], optional
            _description_, by default None
        """
        super().__init__(root, transform)
        self.default_train = default_train
        self.transform = transform
        if not self.default_train:
            self.split = split
        self.download()
        self.process()
        self.init_splits()


    @property
    def raw_file_names(self) -> Tuple[str]:
        return ("data.jsonl", "train.txt", "test.txt", "valid.txt")
    
    @property
    def processed_file_names(self) -> str:
        return "data.pt"
    
    @property
    def num_classes(self) -> int:
        return 2
    
    def process(self) -> None:
        data = read_jsonl(os.path.join(self.root, self.raw_file_names[0]))
        data = {d["idx"]:d["func"] for d in data}

        def _transform(line: str) -> List[Union[str, int]]:
            idx1, idx2, label = line.strip().split('\t')
            return [idx1, idx2, label]

        for file in self.raw_file_names[1:]:
            attr_name = file.split(".")[0]
            attr_name_data = list(read_file(os.path.join(self.root, file), transform=_transform))
            attr_name_data = list(
                                map(lambda x: BCBDatapoint(data[x[0]], data[x[1]], int(x[2])), attr_name_data)
                                )
            setattr(self, attr_name, attr_name_data)
    
    def download(self) -> str:
        for file_name in self.raw_file_names:
            download_url(self.url + file_name, self.root)

    def init_splits(self) -> None:
        if self.default_train:
            self.train = DatapointDataset(datapoints=self.train, transform=self.transform)
            self.test = DatapointDataset(datapoints=self.test, transform=self.transform)
            self.valid = DatapointDataset(datapoints=self.valid, transform=self.transform)        



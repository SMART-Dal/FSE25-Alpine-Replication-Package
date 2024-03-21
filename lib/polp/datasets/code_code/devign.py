import os

from typing import Callable, Optional, Tuple, Dict
from polp.datasets import AbstractDataset, DatapointDataset, download_url, download_gdrive_url
from polp.utils.io import read_json, read_file

class Devign(AbstractDataset):
    
    gid = "1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF"
    url = "https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/Code-Code/Defect-detection/dataset/"

    def __init__(self, 
                 root: str, 
                 default_train: Optional[bool] = True,
                 split: Optional[Tuple[int]] = (80, 10, 10),
                 transform: Optional[Callable] = None,) -> None:
        """Initiate Devign dataset from the `"Devign: Effective Vulnerability Identification 
        by Learning Comprehensive Program Semantics via Graph Neural Networks" <https://arxiv.org/abs/1909.03496>`_ paper.

        Parameters
        ----------
        root : str
            Directory where the dataset will be saved
        default_train : Optional[bool], optional
            If True, default train split is used, otherwise the train/test/val splits are randomly created, by default True
        split : Optional[Tuple[int]], optional
            Tuple of integers representing the percentage of the dataset to be used for the train, test and validation splits, respectively, by default (80, 10, 10)
        transform : Optional[Callable], optional
            _description_, by default None
        """
        super().__init__(root, transform)
        self.default_train = default_train
        if not self.default_train:
            self.split = split
        self.download()
        self.process()
        self.init_splits()

    @property
    def raw_file_names(self) -> Tuple[str]:
        return ("functions.jsonl", "train.txt", "test.txt", "valid.txt")
    
    @property
    def processed_file_names(self) -> str:
        return "data.pt"
    
    @property
    def num_classes(self) -> int:
        return 2

    def download(self) -> str:
        download_gdrive_url(self.gid, self.root, filename=self.raw_file_names[0])
        for file_name in self.raw_file_names[1:]:
            download_url(self.url + file_name, self.root)

    def process(self) -> None:
        def _raw_data_transform(line: Dict) -> Dict:
            return {'function_body': line['func'], 'label':line['target']}
        
        data = read_json(os.path.join(self.root, self.raw_file_names[0]))
        data = list(map(_raw_data_transform, data))

        def _transform(line: str) -> int:
            return int(line.strip())
        
        for file in self.raw_file_names[1:]:
            attr_name = file.replace(".txt", "")
            samples_ids = list(read_file(os.path.join(self.root, file), transform=_transform))
            samples = [data[i] for i in samples_ids]
            setattr(self, attr_name, samples)
        

    def init_splits(self) -> None:
        if self.default_train:
            # The self.* created in L63 by setattr() in process()
            self.train = DatapointDataset(self.train)
            self.test = DatapointDataset(self.test)
            self.valid = DatapointDataset(self.valid)
    
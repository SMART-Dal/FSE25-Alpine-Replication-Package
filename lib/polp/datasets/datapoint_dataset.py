from typing import List, Any, Optional, Callable

class DatapointDataset:

    def __init__(self, 
                 datapoints: List[Any], 
                 prefetch: Optional[bool] = False, 
                 transform: Optional[Callable[[Any], Any]] = None
                 ) -> None:
        """Needed to use pytorch's DataLoader class

        Parameters
        ----------
        datapoints : List[Any]
            List of items to be used in the dataloader. Can be list of paths of files, in-memory objects, etc.
        prefetch : Optional[bool], optional
            This is used when datapoints are stored separetly on disk. It indicates if all datapoints should be read in-memory beforehand, by default False. 
        """
        self.datapoints = datapoints
        self.prefetch = prefetch
        self.transform = transform

    
    def __getitem__(self, index: int) -> Any:
        """Get a sample from the datapoints. Required by pytorch's DataLoader class.

        Parameters
        ----------
        index : int
            Index of the sample to get.

        Returns
        -------
        Any
            Sample from the dataset.
        """
        if self.transform:
            #TODO: Mention somewhere to the user that the transform function should return
            # an object with an asdict method.
            return self.transform(self.datapoints[index]).asdict()
        if type(self.datapoints[index]) == dict:
            return self.datapoints[index]
        return self.datapoints[index].asdict()
    
    def __len__(self) -> int:
        """Get the size of the dataset. Required by pytorch's DataLoader class.

        Returns
        -------
        int
            Size of the dataset.
        """
        return len(self.datapoints)
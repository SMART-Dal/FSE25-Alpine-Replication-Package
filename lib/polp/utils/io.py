import os
import json
from typing import Optional, Callable, Dict, Any

def check_file_exists(path: str) -> bool:
    """Check if a file exists.
    
        Parameters
        ----------
        path : str
            Path to the file.
    
        Returns
        -------
        bool
            True if the file exists, False otherwise.
        """
    return os.path.isfile(path)

def read_file(path: str, transform: Optional[Callable] = None) -> Any:
    """Read a file and return its content.
    
        Parameters
        ----------
        path : str
            Path to the file.
        transform : Optional[Callable], optional
            A function to be applied each line of the file, by default None

        Returns
        -------
        str
            Content of the file.
        """
    with open(path, 'r') as f:
        for line in f:
            if transform is not None:
                yield transform(line)
            else:
                yield line

def read_json(path: str, transform: Optional[Callable] = None) -> Dict:
    """Read a json file and return its content.
    
        Parameters
        ----------
        path : str
            Path to the file.
        transform : Optional[Callable], optional
            A function to be applied on the JSON files content, by default None
    
        Returns
        -------
        str
            Content of the file.
        """
    with open(path, 'r') as f:
        if transform is not None:
            return transform(json.load(f))
        else:
            return json.load(f)

def read_jsonl(path: str, transform: Optional[Callable] = None) -> str:
    """Read a jsonl file and return its content.
    
        Parameters
        ----------
        path : str
            Path to the file.
        transform : Optional[Callable], optional
            A function to be applied each line of the file, by default None
    
        Returns
        -------
        str
            Content of the file.
        """
    lines = []
    with open(path, 'r') as f:
        for line in f:
            if transform is not None:
                lines.append(transform(json.loads(line)))
            else:
                lines.append(json.loads(line))
    return lines
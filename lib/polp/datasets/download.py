from typing import Optional
from polp.utils.io import check_file_exists

import os
import ssl
import urllib
import fsspec

def download_url(
        url: str, 
        folder: str, 
        force_download: Optional[bool] = False,
        filename: Optional[str] = None) -> str:
    """Downalods a file from a given url and place it in root. Adapted from torch_geometric.data.download.download_url.

    Parameters
    ----------
    url : str
        URL of the remote file.
    folder : str
        Root directory where the file is saved.
    force_download : bool, optional
        If True, re-download the file even if it already exists, by default False
    filename : Optional[str], optional
        Filename to be set after completing download. If set to None, the name given by the URL will be used, by default None

    Returns
    -------
    str
        Path to the downloaded file.
    """
    if filename is None:
        filename = url.split('/')[-1]

    path = os.path.join(folder, filename)

    if check_file_exists(path) and not force_download:
        return path

    os.makedirs(folder, exist_ok=True)

    print(f'Downloading {url}')

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url)

    with fsspec.open(path, 'wb') as f:
        # workaround for https://bugs.python.org/issue42853
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path

def download_gdrive_url(
        file_id: str,
        folder: str,
        force_download: Optional[bool] = False,
        filename: Optional[str] = None) -> str:
    """Downloads a file from Google Drive and place it in root. Currently, it does not handle large files.

    Parameters
    ----------
    file_id : str
        Google Drive ID of the file.
    folder : str
        Root directory where the file is saved.
    force_download : Optional[bool], optional
        If True, re-download the file even if it already exists, by default False
    filename : Optional[str], optional
        Filename to be set after completing download. If set to None, the name given by the URL will be used, by default None

    Returns
    -------
    str
        Path to the downloaded file.
    """
    url = f"https://drive.google.com/uc?id={file_id}&export=download"

    if filename is None:
        filename = file_id

    path = os.path.join(folder, filename)

    if check_file_exists(path) and not force_download:
        return path
    
    os.makedirs(folder, exist_ok=True)

    print(f'Downloading GDrive File {file_id}')

    response = urllib.request.urlopen(url)

    file_data = response.read()

    with open(path, 'wb') as file:
        file.write(file_data)

    return path

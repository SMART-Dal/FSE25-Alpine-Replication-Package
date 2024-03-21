import os, random, torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import json

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
def nanstd(o,dim):
    return torch.sqrt(
                torch.nanmean(
                    torch.pow( torch.abs(o-torch.nanmean(o,dim=dim).unsqueeze(dim)),2),
                    dim=dim)
                )

def repack_tensor_and_create_mask(tensor, mask, fuse = False):
    """
    Given a `mask`, this function removes from `tensor` the tokens according to that mask and returns
    the new batch tensor and updated mask.
    If `fuse` is True, it will merge the masked tokens into one tensor which will be included in the new sequence.
    """
    batch = []
    lengths = []
    for el, msk in zip(tensor, mask):
        new_len = msk.sum().item()
        if fuse:
          new_len += 1
        _, hidden_dim = el.shape
        _m = msk[..., None].bool()
        if fuse:
          new_el = el.masked_select(_m)
          inv_m = ~_m
          num_masked_tokens = inv_m.int().sum().item()
          fused_tokens = el.masked_select(inv_m).reshape((num_masked_tokens, hidden_dim)).mean(0)
          new_el = torch.cat((new_el, fused_tokens)).reshape((new_len, hidden_dim))
        else:
          new_el = el.masked_select(_m).reshape((new_len, hidden_dim))
        batch.append(new_el)
        lengths.append(new_len)
    
    padded_batch = pad_sequence(batch, batch_first=True)
    new_mask = (padded_batch > 0).any(-1)

    return padded_batch, new_mask


def save_dict_as_json(data, file_path):
    """
    Saves a given dictionary as a JSON file.

    Parameters:
    - data (dict): The dictionary to save as JSON.
    - file_path (str): The file path where the JSON file will be saved.

    Returns:
    - None
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        print(f"Dictionary saved successfully as JSON in {file_path}")
    except Exception as e:
        print(f"An error occurred while saving the dictionary as JSON: {e}")
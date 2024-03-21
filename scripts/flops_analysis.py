# Adapted from: https://github.com/Mxbonn/ltmp/blob/bb5dc451d296f58c76161cf5173e5c86b2ae93cc/ltmp/utils.py


from polp.nn.layers.attention import AttentionMask
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
from tqdm import tqdm

import os
import torch
import argparse
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from pruning.utils import set_seed
from torch.utils.data import DataLoader, Subset
from transformers import  AutoTokenizer
from pruning.PrunableModel import PrunableModel
from polp.datasets.code_code import Devign, BigCloneBench
from torch.utils.data import DataLoader
from pruning.bcb_classifier import Model as BCBModel
from pruning.classifier import Model as DModel
from pruning.metric_logger import MetricLogger
from pruning.utils import save_dict_as_json
import re
import math, random

metric_logger = MetricLogger()
def init_logger(args):
    # Add seed value, method, and current epoch (initially 0)
    metric_logger.create('alpha', args.alpha)
    metric_logger.create('prunable_layers', args.layers)


def mha_ffn_break_down(flops_dict):
    ffn_pattern = re.compile(r"encoder\.layers\.\d+\.ffn\..+")
    mha_pattern = re.compile(r"encoder\.layers\.\d+\.mha\..+")

    ffn_keys = [key for key in flops_dict.keys() if ffn_pattern.match(key)]
    mha_keys = [key for key in flops_dict.keys() if mha_pattern.match(key)]

    ffn_flops = 0
    mha_flops = 0

    for ffnk in ffn_keys:
        ffn_flops += flops_dict[ffnk]
    
    for mhak in mha_keys:
        mha_flops += flops_dict[mhak]

    return {'FFN': ffn_flops, 'MHA': mha_flops}

def increment_dict(d1, d2):
    for key in d2:
        if key in d1:
            d1[key] += d2[key]
        else:
            raise KeyError(f"Key {key} not found in d1.")
    return d1

def benchmark_flops_with_real_data(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    tokenizer: AutoTokenizer,
    device: torch.device = 0,
    task: str = None,
    verbose: bool = False,
) -> float:
    if not isinstance(device, torch.device):
        device = torch.device(device)

    model = model.eval().to(device)

    if hasattr(data_loader, "batch_size"):
        batch_size = data_loader.batch_size
    else:
        batch_size = data_loader.loader.batch_size

    total_flops = 0
    processed_samples = 0

    initial_dict = None

    with torch.no_grad():
        pbar = tqdm(enumerate(data_loader), total=len(data_loader))
        for batch_idx, batch in pbar:
            if task == "defect_pred":
                input_sentence = tokenizer(batch['function_body'], padding='longest', truncation=True, max_length=512)
                code_input = torch.tensor(input_sentence['input_ids'], device=device)
                code_input = code_input.to(device)
                mask = torch.tensor(input_sentence['attention_mask'], device=device).bool()
                mask = AttentionMask(mask)
            else:
                fn1, fn2, labels = batch['func1'], batch["func2"], batch['label']
                fn1 = tokenizer(fn1, padding='max_length', truncation=True, return_tensors="pt", max_length=512)
                fn2 = tokenizer(fn2, padding='max_length', truncation=True, return_tensors="pt", max_length=512)
                code_input = torch.cat((fn1['input_ids'], fn2['input_ids']), dim=1)
                code_input = code_input.to(device)
                mask = torch.cat((fn1['attention_mask'], fn2['attention_mask']), dim=1).bool()
                mask = mask.to(device)
            
            input = (code_input, mask)
            flops = FlopCountAnalysis(model, input)
            flops.unsupported_ops_warnings(False)
            if initial_dict is None:
                initial_dict = flops.by_module()
            else:
                initial_dict = increment_dict(initial_dict, flops.by_module())
            total_flops += flops.total()
            processed_samples += batch_size
            pbar.set_postfix({"mean FLOPs": f"{total_flops / processed_samples:,.0f}"})

    flops = total_flops / processed_samples

    flops = flops / 1e9 # To GFLOPs

    # Normalize dict
    for k in initial_dict:
        initial_dict[k] = initial_dict[k] / processed_samples
        initial_dict[k] = initial_dict[k] / 1e9 # To GFLOPs

    if verbose:
        print(f"FLOPS: {flops:,}")

    return flops, initial_dict

prunable_layers = {
    'all': [i for i in range(12)],
    'odd_idx': [i for i in range(12) if i%2 == 1],
    'even_idx': [i for i in range(12) if i%2 == 0],
    'none': [],
}

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", default="../", type=str,
                        help="Model checkpoint.")
    parser.add_argument("--task", type=str,
                        help="SE task name." , choices=['defect_pred', 'code_clone'])
    parser.add_argument("--model_name", type=str,
                        help="The model tag such `microsoft/codebert-base`")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--device", default='cuda', type=str,
                        help="Device to be used during analysis: `cuda` or `cpu`.")
    
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint)
    args.alpha = checkpoint['alpha']
    args.layers = checkpoint['layers']
    init_logger(args)
    set_seed(123456)

    if args.task == "code_clone":
        frac=0.01
        # Prepare dataset and dataloaders
        dataset = BigCloneBench(root='./data/bcb', transform=None)
        dataset = dataset.test
        test_indices = [i for i in range(len(dataset))]
        test_indices = random.sample(test_indices, math.floor(frac * len(dataset)))
        dataset = Subset(dataset, test_indices)
    else:
        dataset = Devign(root='./data/devign', transform=None)
        dataset = dataset.test
    
    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    encoder = PrunableModel.from_hf_hub(
        name=args.model_name,
        revision="main",
        output_attention_probs=True,
    )

    if args.task == "code_clone":
        model = BCBModel(encoder=encoder, block_size=512)
    else:
        model = DModel(encoder=encoder)

    model.load_state_dict(checkpoint['model_state_dict'])
    flops, flops_dict = benchmark_flops_with_real_data(
        model=model,
        data_loader=dataloader,
        tokenizer=tokenizer,
        device=device,
        task=args.task,
        verbose=True,
    )
    # Save FLOPS breakdown
    task = args.task
    parent_dir = os.path.dirname(args.checkpoint)
    parent_folder_name = os.path.basename(parent_dir)
    output = os.path.join(os.getcwd(), f"{parent_folder_name}_flops_breakdownn_{task}.json")
    mha_ffn_flops = mha_ffn_break_down(flops_dict)
    save_dict_as_json(dict(flops_dict), output)
    output = os.path.join(os.getcwd(), f"{parent_folder_name}_flops_mha_ffn_{task}.json")
    save_dict_as_json(mha_ffn_flops, output)
if __name__ == "__main__":
    main()

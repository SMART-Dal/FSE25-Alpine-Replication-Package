from polp.nn.layers.attention import AttentionMask
import torch
from tqdm import tqdm
import argparse

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
import math, random, time

metric_logger = MetricLogger()
def init_logger(args):
    # Add seed value, method, and current epoch (initially 0)
    metric_logger.create('alpha', args.alpha)
    metric_logger.create('prunable_layers', args.layers)


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
    
    # Prepare dataset and dataloaders
    if args.task == "code_clone":
        frac=0.01
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

    model.to(device)
    model.eval()
    # Warmup stage
    print("Warming up the model")
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for _, batch in pbar:
            if args.task == "defect_pred":
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
                _ = model(input_ids=code_input, attention_mask=mask)
                break  # Just do one batch for warm-up   

    trials = []
    N = 5
    for i in range(N):
        # Measure throughput
        print(f"Throughput calculation, trial: {i}")
        num_samples = 0
        start_time = time.time()

        with torch.no_grad():
            pbar = tqdm(enumerate(dataloader), total=len(dataloader))
            for _, batch in pbar:
                if args.task == "defect_pred":
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
                    
                _ = model(input_ids=code_input, attention_mask=mask)
                batch_size = code_input.size(0)
                num_samples += batch_size
    
        torch.cuda.synchronize()
        end_time = time.time()
        total_time = end_time - start_time
        throughput = num_samples / total_time

        trials.append(throughput)

        print(f"Processed {num_samples} samples in {total_time:.2f} seconds.")
        print(f"Throughput: {throughput:.2f} samples/sec")

    throughput = sum(trials) / N
    print(f"Avg. throughput: {throughput:.2f} samples/sec")


if __name__ == "__main__":
    main()
import os
import time
import torch
import logging
import argparse
import warnings
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from pruning.utils import set_seed, save_dict_as_json
from sklearn.metrics import recall_score, precision_score, f1_score
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from pruning.PrunableModel import PrunableModel
from polp.datasets.code_code import BigCloneBench
from polp.datasets.code_code.bigclonebenchmark import BCBDatapoint
from torch.utils.data import DataLoader, Subset
from polp.nn.layers import AttentionMask
from pruning.bcb_classifier import Model
from pruning.metric_logger import MetricLogger
import random, math

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
metric_logger = MetricLogger()

prunable_layers = {
    'all': [i for i in range(12)],
    'odd_idx': [i for i in range(12) if i%2 == 1],
    'even_idx': [i for i in range(12) if i%2 == 0],
    'none': [],
}

def init_logger(args):
    # Add seed value, method, and current epoch (initially 0)
    metric_logger.create('seed', args.seed)
    metric_logger.create('alpha', args.alpha)
    metric_logger.create('collection_mode', False)
    full_folder = f"checkpoint_alpha_{args.alpha}_{args.seed}"
    metric_logger.create('prunable_layers', prunable_layers[args.layers])
    metric_logger.create('save_dir', os.path.join(args.output_dir, full_folder))
    metric_logger.create('val_batch_idx', 0)
    metric_logger.create('current_epoch', 0)

        

def train(args, model, train_dataloader, val_dataloader, tokenizer):
    
    
    len_dl = len(train_dataloader)
    #len_dl = 2 # Did this for testing
    args.max_steps = args.epoch*len_dl
    args.save_steps = len_dl
    args.warmup_steps = len_dl
    args.logging_steps = len_dl
    args.num_train_epochs = args.epoch
    
    model.to(args.device)
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters(
        ) if not any(nd in n for nd in no_decay)]}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1,
                                                num_training_steps=args.max_steps)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(args.train_dp))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_acc = 0
    model.zero_grad()
    
    model.train()
    for idx in range(0, int(args.num_train_epochs)):
        
        metric_logger.update('current_epoch', idx)
        
        # This is for test. Remove later.
        
        #two_batches = [next(iter(train_dataloader)), next(iter(train_dataloader))]
        
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        #bar = tqdm(two_batches, total=len(two_batches))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            fn1, fn2, labels = batch['func1'], batch["func2"], batch['label']
            fn1 = tokenizer(fn1, padding='max_length', truncation=True, return_tensors="pt", max_length=512)
            fn2 = tokenizer(fn2, padding='max_length', truncation=True, return_tensors="pt", max_length=512)
            code_input = torch.cat((fn1['input_ids'], fn2['input_ids']), dim=1)
            code_input = code_input.to(args.device)
            mask = torch.cat((fn1['attention_mask'], fn2['attention_mask']), dim=1).bool()
            mask = mask.to(args.device)
            labels = torch.tensor(labels)
            labels = labels.to(args.device)
            
            loss, _ = model(input_ids=code_input, attention_mask=mask, labels=labels)
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss/tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                avg_loss = round(
                    np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb = global_step

                if args.save_steps > 0 and global_step % args.save_steps == 0:

                    if args.evaluate_during_training:
                        results = evaluate(args, model, val_dataloader, tokenizer, eval_when_training=True)

                    logger.info("  "+"*"*20)
                    logger.info("  Current ACC:%s", round(results["eval_acc"], 4))
                    logger.info("  Best ACC:%s", round(best_acc, 4))
                    logger.info("  "+"*"*20)

                    if results["eval_acc"] >= best_acc:
                        best_acc = results["eval_acc"]
                        model_name = args.model_name.split("/")[1]
                        checkpoint_prefix = f"bcb_{model_name}_checkpoint_alpha_{args.alpha}_layers_{args.layers}_{args.seed}"
                        output_dir = os.path.join(args.output_dir, "{}".format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)

                        output_dir = os.path.join(output_dir, "{}".format("model.bin"))
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'alpha': args.alpha,
                            'layers': metric_logger.get('prunable_layers')
                        }, output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                    else:
                        logger.info("Model checkpoint are not saved")

    return results


def evaluate(args, model, eval_dataloader, tokenizer, eval_when_training=False):

    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(args.val_dp))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    logits = []
    labels = []

    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    batch_counter = 0
    for batch in bar:
        metric_logger.update('val_batch_idx', batch_counter)
        bar.set_description("evaluation")
        fn1, fn2, _labels = batch['func1'], batch["func2"], batch['label']
        fn1 = tokenizer(fn1, padding='max_length', truncation=True, return_tensors="pt", max_length=512)
        fn2 = tokenizer(fn2, padding='max_length', truncation=True, return_tensors="pt", max_length=512)
        code_input = torch.cat((fn1['input_ids'], fn2['input_ids']), dim=1)
        code_input = code_input.to(args.device)
        mask = torch.cat((fn1['attention_mask'], fn2['attention_mask']), dim=1).bool()
        mask = mask.to(args.device)
        _labels = torch.tensor(_labels)
        _labels = _labels.to(args.device)
        with torch.no_grad():
            logit = model(input_ids=code_input, attention_mask=mask)
            logits.append(logit.cpu().numpy())
        labels.append(_labels.cpu().numpy())
        batch_counter += 1
        
        

    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    # logits = F.sigmoid(torch.FloatTensor(logits))
    preds = logits[:, 0] > 0.5
    eval_acc = np.mean(labels==preds)
    recall = recall_score(labels, preds)
    precision = precision_score(labels, preds)
    f1 = f1_score(labels, preds)
    result = {
        "eval_acc": eval_acc,
        "eval_precision": float(precision),
        "eval_recall": float(recall),
        "eval_f1": float(f1)
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def main():
    
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default="../", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--alpha", default=1., type=float,
                        help="How far from the mean should token scores be.")
    parser.add_argument("--layers", default='all', type=str,
                        help="Layer indices where pruning should occur. Options: all, even_idx, odd_idx")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--model_name", type=str,
                        help="The model tag such `microsoft/codebert-base`")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--epoch", type=int, default=5,
                        help="Number of epochs")

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    args = parser.parse_args()
    logger.info(args)
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.per_gpu_train_batch_size = args.train_batch_size//args.n_gpu
    args.per_gpu_eval_batch_size = args.eval_batch_size//args.n_gpu

    logger.info("Device: %s, n_gpu: %s", args.device, args.n_gpu)

    set_seed(args.seed)    
    init_logger(args)


    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    encoder = PrunableModel.from_hf_hub(
        name=args.model_name,
        revision="main",
        output_attention_probs=True,
    )

    # Prepare dataset and dataloaders
    dataset = BigCloneBench(root='./data/bcb', transform=None)
    args.train_dp, args.val_dp = dataset.train, dataset.test

    model = Model(encoder=encoder, block_size=512)
    
    
    logger.info("Training/evaluation parameters %s", args)

    train_indices = [i for i in range(len(args.train_dp))]
    test_indices = [i for i in range(len(args.val_dp))]
    # Take a random sample of size math.floor(fract * len(._indices)). When seed is fixed, this will be the same.
    frac = 0.1
    train_indices = random.sample(train_indices, math.floor(frac * len(args.train_dp)))
    test_indices = random.sample(test_indices, math.floor(0.1 * frac * len(args.val_dp)))

    train_dataset = Subset(args.train_dp, train_indices)
    test_dataset = Subset(args.val_dp, test_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)
    eval_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, pin_memory=True)

    if args.do_train:
        start_time = time.time()
        results = train(args, model, train_dataloader, eval_dataloader, tokenizer)
        torch.cuda.synchronize()
        end_time = time.time()
        total_taining_time = end_time - start_time
        results['training_time'] = total_taining_time
        model_name = args.model_name.split("/")[1]
        results_prefix = f"bcb_{model_name}_alpha_{args.alpha}_{args.seed}_layers_{args.layers}_results.json"
        output_dir = os.path.join(args.output_dir, "{}".format(results_prefix))
        save_dict_as_json(results, results_prefix)

    if args.do_eval:
        checkpoint_prefix = "checkpoint/model.bin"
        output_dir = os.path.join(args.output_dir, "{}".format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        evaluate(args, model, tokenizer)


if __name__ == "__main__":
    main()

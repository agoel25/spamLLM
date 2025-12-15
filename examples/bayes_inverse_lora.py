#!/usr/bin/env python3
"""lora fine-tuning trianing script"""
"""Attributions: 
- file majorly adapted from other files (especially bayes_inverse.py) that were provided in this project
"""

import os
import torch
from torch.utils.data import DataLoader
from dotenv import load_dotenv
from tqdm import tqdm
import argparse

from autograder.dataset import CPEN455_2025_W1_Dataset, prepare_subset
from model import LlamaModel
from model.lora import apply_lora_to_model, get_lora_parameters
from utils.weight_utils import load_model_weights
from model.config import Config
from model.tokenizer import Tokenizer
from utils.download import _resolve_snapshot_path
from utils.device import set_device
from examples.bayes_inverse import get_seq_log_prob, bayes_inverse_llm_classifier, save_probs, ENRON_LABEL_INDEX_MAP
from utils.prompt_template import get_prompt


def train_lora(args, model, tokenizer, train_loader, val_loader, device):
    optimizer = torch.optim.AdamW(get_lora_parameters(model), lr=args.learning_rate)
    
    best_val_acc = 0
    for epoch in range(args.num_epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            _, subjects, messages, label_indices = batch
            
            # convert label indices back to text
            labels_text = [ENRON_LABEL_INDEX_MAP.inv[int(l)] for l in label_indices]

            # build prompts with ground turth lables 
            prompts = [get_prompt(subject=s, message=m, label=l, max_seq_length=args.max_seq_len) 
                       for s, m, l in zip(subjects, messages, labels_text)]
            
            # compute log prob under the model
            seq_log_prob = get_seq_log_prob(prompts, tokenizer, model, device)

            # negative log likelihood 
            num_chars = torch.tensor([len(p) for p in prompts], device=device).sum()
            loss = -seq_log_prob.sum() / num_chars
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # validation check for classification accuracy
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                is_correct, _ = bayes_inverse_llm_classifier(args, model, batch, tokenizer, device)
                if is_correct is not None:
                    correct += is_correct.sum().item()
                    total += len(is_correct)
        
        val_acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}: Val Accuracy = {val_acc:.2%}")
        
        # check if validation accuracy improved, if it did, save the checkpoint
        # note we only save lora weights here, not the frozen weights
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'lora_state_dict': {k: v for k, v in model.state_dict().items() if 'lora' in k}
            }, 'examples/ckpts/lora_best.pt')
    
    print(f"Best validation accuracy: {best_val_acc:.2%}")
    return model


if __name__ == "__main__":
    torch.manual_seed(42)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--dataset_path", type=str, default="autograder/cpen455_released_datasets/train_val_subset.csv")
    parser.add_argument("--test_dataset_path", type=str, default="autograder/cpen455_released_datasets/test_subset.csv")
    parser.add_argument("--prob_output_folder", type=str, default="bayes_inverse_probs")
    parser.add_argument("--user_prompt", type=str, default="")
    args = parser.parse_args()
    
    load_dotenv()
    device = set_device()
    
    checkpoint = os.getenv("MODEL_CHECKPOINT")
    model_cache_dir = os.getenv("MODEL_CACHE_DIR")
    
    # load model and tokenizer
    tokenizer = Tokenizer.from_pretrained(checkpoint, cache_dir=model_cache_dir)
    base_path = _resolve_snapshot_path(checkpoint, cache_dir=model_cache_dir)
    config = Config._find_config_files(base_path)
    
    model = LlamaModel(config)
    load_model_weights(model, checkpoint, cache_dir=model_cache_dir, device=device)
    
    # apply LoRA
    model = apply_lora_to_model(model, r=args.lora_r, alpha=args.lora_alpha)
    model = model.to(device)
    
    # prepare data
    train_val_dataset = CPEN455_2025_W1_Dataset(csv_path=args.dataset_path)
    train_dataset, val_dataset = prepare_subset(
        train_val_dataset, int(0.8 * len(train_val_dataset)), ratio_spam=0.5, return_remaining=True
    )
    test_dataset = CPEN455_2025_W1_Dataset(csv_path=args.test_dataset_path)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    os.makedirs(args.prob_output_folder, exist_ok=True)
    os.makedirs("examples/ckpts", exist_ok=True)
    
    # train
    model = train_lora(args, model, tokenizer, train_loader, val_loader, device)
    
    # save probabilities
    train_val_loader = DataLoader(train_val_dataset, batch_size=args.batch_size, shuffle=False)
    save_probs(args, model, tokenizer, train_val_loader, device=device, name="train_n_val")
    save_probs(args, model, tokenizer, test_loader, device=device, name="test")
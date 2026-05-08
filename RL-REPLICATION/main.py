"""Main entrypoint for replicating the RR task from IL-TUR benchmark."""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader import create_label_maps, encode_labels, load_rr_dataset
from dataset import RRDataset, load_tokenizer, tokenize_function
from evaluate import evaluate_model
from model import initialize_model
from train import train_model


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model) -> int:
    """Count total trainable and non-trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def parse_args():
    """Parse command-line arguments for RR training pipeline."""
    parser = argparse.ArgumentParser(description="Train and evaluate RR classifier.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to train JSON file")
    parser.add_argument("--dev_file", type=str, required=True, help="Path to dev JSON file")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test JSON file")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--checkpoint_path", type=str, default="best_rr_model.pt")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Run end-to-end RR pipeline: load data, train model, and evaluate on test set."""
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Load train/dev/test data from RR JSON files.
    train_examples = load_rr_dataset(args.train_file)
    dev_examples = load_rr_dataset(args.dev_file)
    test_examples = load_rr_dataset(args.test_file)

    # 2) Build fixed label maps for the 13 RR classes.
    label2id, id2label = create_label_maps()

    # 3) Encode labels and tokenize all splits.
    train_encoded = encode_labels(train_examples, label2id)
    dev_encoded = encode_labels(dev_examples, label2id)
    test_encoded = encode_labels(test_examples, label2id)

    tokenizer, tokenizer_name = load_tokenizer()

    tokenized_train = tokenize_function(train_encoded, tokenizer, max_length=args.max_seq_length)
    tokenized_dev = tokenize_function(dev_encoded, tokenizer, max_length=args.max_seq_length)
    tokenized_test = tokenize_function(test_encoded, tokenizer, max_length=args.max_seq_length)

    # 4) Create PyTorch datasets and dataloaders.
    train_dataset = RRDataset(tokenized_train)
    dev_dataset = RRDataset(tokenized_dev)
    test_dataset = RRDataset(tokenized_test)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)

    # 5) Initialize model (with fallback if InLegalBERT is unavailable).
    model, model_name = initialize_model(label2id=label2id, id2label=id2label)
    model.to(device)

    # 6) Log hyperparameters and model size for reproducibility.
    print("Hyperparameters:")
    print(f"  model: {model_name}")
    print(f"  tokenizer: {tokenizer_name}")
    print(f"  max_seq_length: {args.max_seq_length}")
    print(f"  train_batch_size: {args.train_batch_size}")
    print(f"  eval_batch_size: {args.eval_batch_size}")
    print(f"  learning_rate: {args.learning_rate}")
    print(f"  epochs: {args.epochs}")
    print(f"  warmup_ratio: {args.warmup_ratio}")
    print(f"  weight_decay: {args.weight_decay}")
    print(f"  seed: {args.seed}")
    print(f"Model parameter count: {count_parameters(model):,}")

    # 7) Train and save best checkpoint by validation macro F1.
    train_stats = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=dev_loader,
        epochs=args.epochs,
        lr=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        checkpoint_path=args.checkpoint_path,
        device=device,
    )
    print(f"Best validation macro F1: {train_stats['best_val_macro_f1']:.4f}")

    # 8) Reload best checkpoint and evaluate on test split.
    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best checkpoint from: {args.checkpoint_path}")

    metrics = evaluate_model(model=model, test_loader=test_loader, id2label=id2label, device=device)

    # 9) Print final macro F1.
    print(f"Final Test Macro F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()

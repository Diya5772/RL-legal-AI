"""Dataset and tokenization utilities for RR classification."""

from __future__ import annotations

from typing import Dict, List, Union

import torch
from transformers import AutoTokenizer


def load_tokenizer(primary_model: str = "law-ai/InLegalBERT", fallback_model: str = "bert-base-uncased"):
    """Load tokenizer with fallback if the primary model tokenizer cannot be downloaded."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(primary_model)
        print(f"Using tokenizer: {primary_model}")
        return tokenizer, primary_model
    except Exception as exc:
        print(f"Failed to load tokenizer '{primary_model}' ({exc}). Falling back to '{fallback_model}'.")
        tokenizer = AutoTokenizer.from_pretrained(fallback_model)
        print(f"Using tokenizer: {fallback_model}")
        return tokenizer, fallback_model


def tokenize_function(
    encoded_examples: List[Dict[str, object]],
    tokenizer,
    max_length: int = 128,
) -> Dict[str, Union[List[int], List[List[int]]]]:
    """Tokenize RR examples into model-ready inputs.

    Args:
        encoded_examples: Label-encoded examples containing "text" and "label_id".
        tokenizer: Hugging Face tokenizer instance.
        max_length: Max sequence length for truncation/padding.

    Returns:
        A dictionary with input_ids, attention_mask, token_type_ids, and labels.
    """
    texts = [str(item["text"]) for item in encoded_examples]
    labels = [int(item["label_id"]) for item in encoded_examples]

    tokenized = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
    )

    # Ensure token_type_ids always exists to keep downstream batch handling simple.
    if "token_type_ids" not in tokenized:
        tokenized["token_type_ids"] = [[0] * len(ids) for ids in tokenized["input_ids"]]

    tokenized["labels"] = labels
    return tokenized


class RRDataset(torch.utils.data.Dataset):
    """PyTorch dataset wrapper for tokenized RR examples."""

    def __init__(self, tokenized_data: Dict[str, List[int]]):
        """Initialize dataset from tokenized data dictionary."""
        self.tokenized_data = tokenized_data

    def __len__(self) -> int:
        """Return number of examples in the dataset."""
        return len(self.tokenized_data["labels"])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get one tokenized example as tensors."""
        return {
            "input_ids": torch.tensor(self.tokenized_data["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.tokenized_data["attention_mask"][idx], dtype=torch.long),
            "token_type_ids": torch.tensor(self.tokenized_data["token_type_ids"][idx], dtype=torch.long),
            "labels": torch.tensor(self.tokenized_data["labels"][idx], dtype=torch.long),
        }

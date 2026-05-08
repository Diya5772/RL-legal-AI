"""Training loop for RR sentence classification."""

from __future__ import annotations

from typing import Dict

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup


def _compute_macro_f1(model, data_loader, device) -> float:
    """Compute macro F1 score on a validation data loader."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return f1_score(all_labels, all_preds, average="macro")


def train_model(
    model,
    train_loader,
    val_loader,
    epochs: int = 5,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    checkpoint_path: str = "best_rr_model.pt",
    device: torch.device | None = None,
) -> Dict[str, float]:
    """Train RR classifier and save best checkpoint by validation macro F1.

    Args:
        model: Sequence classification model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        epochs: Number of training epochs.
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        warmup_ratio: Fraction of total steps used for warmup.
        checkpoint_path: Where to save best model checkpoint.
        device: Torch device (auto-selects CUDA if available).

    Returns:
        Dict with best validation macro F1.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = max(1, len(train_loader) * epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_macro_f1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} - Training")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs.logits
            loss = criterion(logits, labels)

            loss.backward()

            # Gradient clipping stabilizes training for transformer fine-tuning.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_loss / max(1, len(train_loader))
        val_macro_f1 = _compute_macro_f1(model, val_loader, device)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Macro F1: {val_macro_f1:.4f}"
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_macro_f1": best_val_macro_f1,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
            print(f"Saved new best checkpoint to: {checkpoint_path}")

    return {"best_val_macro_f1": best_val_macro_f1}

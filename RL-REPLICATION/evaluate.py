"""Evaluation utilities for RR sentence classification."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm


def evaluate_model(model, test_loader, id2label, device: torch.device | None = None) -> Dict[str, object]:
    """Evaluate model on test set and return aggregate and per-class metrics.

    Args:
        model: Trained sequence classification model.
        test_loader: Test DataLoader.
        id2label: Mapping from class id to label string.
        device: Torch device (auto-selects CUDA if available).

    Returns:
        Dict containing macro F1, classification report dict, and confusion matrix.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            preds = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    target_names = [id2label[idx] for idx in sorted(id2label.keys())]
    label_ids = sorted(id2label.keys())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    report_text = classification_report(
        all_labels,
        all_preds,
        labels=label_ids,
        target_names=target_names,
        zero_division=0,
    )
    report_dict = classification_report(
        all_labels,
        all_preds,
        labels=label_ids,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=label_ids)

    print(f"Test Macro F1: {macro_f1:.4f}")
    print("Per-class F1 report:")
    print(report_text)
    print("Confusion Matrix:")
    print(cm)

    # Optional plotting if matplotlib is available in the environment.
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(len(target_names)),
            yticks=np.arange(len(target_names)),
            xticklabels=target_names,
            yticklabels=target_names,
            ylabel="True label",
            xlabel="Predicted label",
            title="RR Confusion Matrix",
        )
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
        fig.tight_layout()
        plt.show()
    except Exception as exc:
        print(f"Skipping confusion matrix plot: {exc}")

    return {
        "macro_f1": macro_f1,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
    }

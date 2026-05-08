"""Model initialization utilities for RR classification."""

from __future__ import annotations

from transformers import AutoModelForSequenceClassification


def initialize_model(
    label2id,
    id2label,
    primary_model: str = "law-ai/InLegalBERT",
    fallback_model: str = "bert-base-uncased",
):
    """Initialize sequence classification model with fallback.

    Args:
        label2id: Label-to-id mapping.
        id2label: Id-to-label mapping.
        primary_model: Preferred pretrained checkpoint.
        fallback_model: Backup checkpoint if primary load fails.

    Returns:
        Tuple of (model, model_name_used).
    """
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            primary_model,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
        )
        print(f"Using model: {primary_model}")
        return model, primary_model
    except Exception as exc:
        print(f"Failed to load model '{primary_model}' ({exc}). Falling back to '{fallback_model}'.")
        model = AutoModelForSequenceClassification.from_pretrained(
            fallback_model,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
        )
        print(f"Using model: {fallback_model}")
        return model, fallback_model

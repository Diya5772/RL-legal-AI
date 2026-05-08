"""Utilities for loading and encoding the Rhetorical Role (RR) dataset."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


RR_LABELS = [
    "Fact",
    "Issue",
    "ArgumentPetitioner",
    "ArgumentRespondent",
    "Statute",
    "Dissent",
    "PrecedentReliedUpon",
    "PrecedentNotReliedUpon",
    "PrecedentOverruled",
    "RulingByLowerCourt",
    "RatioOfTheDecision",
    "RulingByPresentCourt",
    "None",
]


def load_rr_dataset(json_file_path: str) -> List[Dict[str, str]]:
    """Load RR dataset JSON and return sentence-label examples.

    Args:
        json_file_path: Absolute or relative path to a split JSON file.

    Returns:
        A list of dictionaries with keys "text" and "label".

    Notes:
        - Skips entries with empty text or missing labels.
        - For multi-label instances, keeps only the first label.
    """
    path = Path(json_file_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    examples: List[Dict[str, str]] = []
    skipped_empty_text = 0
    skipped_missing_label = 0

    for case in data:
        annotations = case.get("annotations", [])
        for annotation in annotations:
            results = annotation.get("result", [])
            for result in results:
                value = result.get("value", {})
                text_list = value.get("text", [])
                labels = value.get("labels", [])

                # Keep first sentence string if provided as a list.
                sentence = text_list[0] if isinstance(text_list, list) and text_list else ""
                sentence = sentence.strip() if isinstance(sentence, str) else ""

                if not sentence:
                    skipped_empty_text += 1
                    continue

                # Keep only first label in case of multi-label annotations.
                label = labels[0] if isinstance(labels, list) and labels else None
                if not label:
                    skipped_missing_label += 1
                    continue

                examples.append({"text": sentence, "label": label})

    print(
        f"Loaded {len(examples)} examples from {json_file_path} "
        f"(skipped empty text: {skipped_empty_text}, skipped missing label: {skipped_missing_label})."
    )
    return examples


def create_label_maps() -> tuple[Dict[str, int], Dict[int, str]]:
    """Create fixed RR label-id mappings for the 13 benchmark labels."""
    label2id = {label: idx for idx, label in enumerate(RR_LABELS)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def encode_labels(examples: List[Dict[str, str]], label2id: Dict[str, int]) -> List[Dict[str, object]]:
    """Encode string labels into integer ids and print class distribution.

    Args:
        examples: List of {"text": str, "label": str} dictionaries.
        label2id: Mapping from label string to integer class id.

    Returns:
        List of {"text": str, "label": str, "label_id": int} dictionaries,
        excluding unknown labels.
    """
    encoded: List[Dict[str, object]] = []
    unknown_labels = 0

    for item in examples:
        label = item["label"]
        if label not in label2id:
            unknown_labels += 1
            continue
        encoded.append(
            {
                "text": item["text"],
                "label": label,
                "label_id": label2id[label],
            }
        )

    distribution = Counter(entry["label"] for entry in encoded)
    print("Class distribution:")
    for label_name, count in sorted(distribution.items(), key=lambda x: x[0]):
        print(f"  {label_name}: {count}")

    if unknown_labels:
        print(f"Skipped {unknown_labels} examples with unknown labels.")

    return encoded

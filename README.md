# RL-legal-AI

A comprehensive repository for research and development in Indian Legal AI, focusing on Rhetorical Role (RR) labeling, Legal Question Answering, and Reasoning tasks using state-of-the-art LLMs and Reinforcement Learning techniques.

## 📂 Project Structure

The repository is organized into several modules and research approaches:

### 1. [RL-REPLICATION](./RL-REPLICATION)
This module focuses on the **Rhetorical Role (RR)** labeling task from the **IL-TUR** benchmark.
- **Model**: `InLegalBERT` (Fine-tuned for legal document segmentation).
- **Task**: Classifying segments of legal documents into 13 rhetorical roles (e.g., Facts, Argument, Ratio, Decision).
- **Key Files**:
  - `main.py`: Entry point for training and evaluation.
  - `train.py` & `evaluate.py`: Core logic for model training and metric calculation.
  - `model.py`: Architecture initialization using `transformers`.

### 2. [Approach 1: TinyLlama for Legal QA](./approach%201)
Fine-tuning the **TinyLlama-1.1B** model on the **IndicLegalQA** dataset.
- **Dataset**: 10,000+ Indian Legal QA pairs.
- **Method**: Parameter-efficient fine-tuning (PEFT) using LoRA.
- **Focus**: Improving the conversational capabilities of small LLMs for specific legal queries.

### 3. [Approach 2: Bail Judgment Analysis](./approach%202)
Exploratory data analysis and structured data generation from the **IndianBailJudgments** dataset.
- **Dataset**: `bail_dataset.csv` (1,200+ cases).
- **Task**: Transforming raw judgment text into structured Step-by-Step QA pairs for training reasoning models.

### 4. [Approach 3: Phi-3 Legal Assistant](./approach%203)
Advanced fine-tuning of **Phi-3-mini-4k-instruct** to create an "Expert Indian Legal Assistant".
- **Technique**: 4-bit quantization (QLoRA) for efficient training on consumer hardware.
- **Prompt Engineering**: Uses structured instruction formats to enforce step-by-step reasoning (Facts -> Law -> Reasoning -> Decision).

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- PyTorch (with CUDA support for training)
- Transformers, PEFT, Accelerate, BitsAndBytes

### Installation
```bash
pip install -r RL-REPLICATION/requirements.txt
# or for notebook environments:
pip install transformers accelerate peft bitsandbytes datasets trl
```

## ⚖️ Datasets Used
- **IL-TUR Benchmark**: Used for Rhetorical Role classification.
- **IndicLegalQA**: A 10K sample dataset for Indian legal question answering.
- **IndianBailJudgments**: A specialized dataset for bail-related legal reasoning.

## 🛠️ Models
- **InLegalBERT**: Specialized BERT for Indian Legal documents.
- **Phi-3-mini**: Lightweight yet powerful instruct model from Microsoft.
- **TinyLlama**: 1.1B parameter model optimized for efficiency.

---
*Developed by Diya Pansheriya*

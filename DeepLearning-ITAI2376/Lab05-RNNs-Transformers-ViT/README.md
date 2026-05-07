# Lab 05 — RNNs vs. Transformers vs. Vision Transformers ⚡

**Course:** ITAI-2376 Deep Learning | Spring 2026  
**Instructor:** Prof. Patricia McManus  
**Author:** Munachimso Benson  
**Type:** Major Lab Notebook (4 Parts)

---

## Problem Statement

Three architectural families define modern deep learning: Recurrent Neural Networks, Transformers, and Vision Transformers. This lab implemented, trained, and benchmarked all three from scratch or via fine-tuning — on both text and image tasks — to build a concrete, data-driven understanding of where each architecture shines and where it falls short.

## Datasets

**AG News** (text classification, 4 categories: World, Sports, Business, Sci/Tech)
- Training: 120,000 samples | Test: 7,600 samples
- Loaded via HuggingFace `datasets` library

**CIFAR-10** (image classification, 10 classes)
- Training: 50,000 images | Test: 10,000 images
- 32×32 RGB images
- Loaded via `torchvision.datasets.CIFAR10`

> Both datasets download automatically when running the notebook.

## Approach & Methodology

### Part A — RNN Variants on AG News

Custom vocabulary of 10,000 words, tokenized and padded to fixed length.

| Model | Architecture | Val Accuracy | Parameters |
|-------|-------------|-------------|-----------|
| Vanilla RNN | Single-direction RNN | 70.3% | 1.35M |
| BiLSTM | Bidirectional LSTM | **81.2%** | 1.55M |
| BiGRU | Bidirectional GRU | 80.95% | 1.48M |

- Vanilla RNN exposed the **vanishing gradient problem** clearly — performance plateaued early
- BiLSTM and BiGRU both overcame this with gating mechanisms, with LSTM edging ahead

### Part B — DistilBERT Fine-tuning on AG News

- Loaded `distilbert-base-uncased` from HuggingFace
- Added classification head, fine-tuned for 3 epochs
- **Result: 91.05% accuracy** — a ~10% jump over the best RNN

```python
from transformers import DistilBertForSequenceClassification
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=4
)
```

### Part C — Vision Transformer (ViT) vs Custom CNN on CIFAR-10

| Model | Accuracy | Parameters | Training Time |
|-------|----------|-----------|--------------|
| SimpleCNN (scratch) | 49.6% | 1.1M | 5 seconds |
| ViT (pretrained, fine-tuned) | **96.60%** | 86M | 892 seconds |

- ViT attention maps visualized — the model correctly focused on aircraft fuselage/wings while ignoring background
- CNN trained from scratch couldn't overcome the data scarcity problem at this scale

### Part D — Master Comparison

| Model | Task | Accuracy | Params | Time |
|-------|------|----------|--------|------|
| Vanilla RNN | Text | 70.3% | 1.35M | Fast |
| BiLSTM | Text | 81.2% | 1.55M | Fast |
| BiGRU | Text | 80.95% | 1.48M | Fast |
| DistilBERT | Text | **91.05%** | 67M | 273s |
| SimpleCNN | Image | 49.6% | 1.1M | 5s |
| ViT | Image | **96.60%** | 86M | 892s |

## Results & Evaluation

- Pretrained Transformers dominate both text and image tasks when sufficient compute is available
- RNNs remain competitive when compute is limited and sequences are short
- CNNs from scratch are fast but require much more data to match pretrained vision models
- Attention visualization confirmed ViT attends to semantically meaningful image regions

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
numpy
matplotlib
pandas
scikit-learn
```

Install with:
```bash
pip install torch torchvision transformers datasets numpy matplotlib pandas scikit-learn
```

> **Note:** GPU strongly recommended. ViT fine-tuning took ~892 seconds on a T4 GPU (Google Colab).

## Files

| File | Description |
|------|-------------|
| `L05_Munachimso_Benson_ITAI2376.ipynb` | Full 4-part notebook with all models, results, and attention maps |

## Learning Outcomes

- Understood why vanilla RNNs fail on long sequences and how LSTM/GRU gates solve it
- Learned how self-attention allows Transformers to capture global context without sequential processing
- Experienced the dramatic accuracy gains that come from pretrained representations
- Practiced fine-tuning large pretrained models (DistilBERT, ViT) for downstream tasks

---

*ITAI-2376 · Spring 2026 · Houston Community College*

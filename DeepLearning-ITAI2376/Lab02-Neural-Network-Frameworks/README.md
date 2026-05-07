# Lab 02 — Neural Network Frameworks: PyTorch vs TensorFlow 🧠

**Course:** ITAI-2376 Deep Learning | Spring 2026  
**Instructor:** Prof. Patricia McManus  
**Author:** Munachimso Benson  
**Type:** Hands-on Lab Notebook

---

## Problem Statement

Two frameworks dominate deep learning: PyTorch and TensorFlow/Keras. This lab required building, training, and evaluating the same neural network in both frameworks on the Fashion-MNIST dataset — then comparing their design philosophies, APIs, and performance to understand when and why you'd choose one over the other.

## Dataset

**Fashion-MNIST** — 70,000 grayscale images (28×28 pixels) across 10 clothing categories:  
T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

- Training samples: 60,000 (split 80/20 for train/val)
- Test samples: 10,000
- Image size: 28×28 pixels, grayscale

> Dataset downloads automatically via `torchvision.datasets.FashionMNIST` or `tensorflow.keras.datasets.fashion_mnist`

## Approach & Methodology

### Model Architecture (same in both frameworks)
```
Input (784)  →  Dense(256, ReLU)  →  Dropout(0.5)  →  Dense(128, ReLU)  →  Dense(10, Softmax)
```

### PyTorch Implementation
- Defined model using `nn.Module` subclass
- Used `DataLoader` with custom `Dataset` class
- Training loop written manually (zero_grad → forward → loss → backward → step)
- Optimizer: Adam | Loss: CrossEntropyLoss

### TensorFlow/Keras Implementation
- Defined model using `Sequential` API
- Used `.fit()` for training — Keras handles the loop
- Optimizer: Adam | Loss: sparse_categorical_crossentropy

### Experiments & Tuning
| Experiment | Change | Result |
|------------|--------|--------|
| Baseline | batch=64, lr=0.001, 10 epochs | ~87% val accuracy |
| Larger batch | batch=128 | Faster per-epoch, similar accuracy |
| Dropout added | Dropout(0.5) | Improved generalization |
| LeakyReLU | Replaced ReLU | ~88.55% val accuracy |
| 12 epochs | Extended training | **~89.14% val accuracy** |

## Results & Evaluation

| Framework | Best Val Accuracy | Notes |
|-----------|------------------|-------|
| PyTorch | ~88.8% | More control, manual loop |
| TensorFlow/Keras | ~89.14% | Faster to prototype |

**Framework Comparison:**

| Aspect | PyTorch | TensorFlow/Keras |
|--------|---------|-----------------|
| API style | Pythonic, explicit | High-level, concise |
| Debugging | Eager execution, easy | Can be abstract |
| Flexibility | Very high | High (with custom layers) |
| Community | Research-dominant | Industry-dominant |

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
tensorflow>=2.12.0
numpy
matplotlib
seaborn
scikit-learn
```

Install with:
```bash
pip install torch torchvision tensorflow numpy matplotlib seaborn scikit-learn
```

## Files

| File | Description |
|------|-------------|
| `Module_02_Lab_02_Munachimso_Benson.ipynb` | Full notebook with code, outputs, and analysis |

## Learning Outcomes

- Built a complete deep learning pipeline from scratch in two major frameworks
- Understood the trade-offs between PyTorch's flexibility and Keras's simplicity
- Learned how Dropout regularization prevents overfitting
- Practiced systematic hyperparameter tuning using batch size, learning rate, and activation functions

---

*ITAI-2376 · Spring 2026 · Houston Community College*

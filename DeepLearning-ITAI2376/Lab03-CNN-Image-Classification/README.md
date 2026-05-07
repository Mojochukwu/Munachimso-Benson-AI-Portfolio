# Lab 03 — CNN Image Classification: Puppy or Bagel? 🐶🥯

**Course:** ITAI-2376 Deep Learning | Spring 2026  
**Instructor:** Prof. Patricia McManus  
**Author:** Munachimso Benson  
**Type:** Hands-on Lab Notebook

---

## Problem Statement

Binary image classification is one of the most common real-world deep learning tasks. This lab used the viral "Puppy or Bagel" internet meme dataset to explore the difference between building a CNN from scratch versus leveraging a powerful pretrained model through transfer learning — and to understand why the gap between the two is so significant.

## Dataset

**Puppy or Bagel** (via Kaggle) — binary image classification  
- **Classes:** Puppy (`1`) vs Bagel (`0`)
- **Format:** RGB images, resized to 224×224
- **Split:** 80% training / 20% validation

> Access: Download from Kaggle at `datasets/biplobdey/puppy-or-bagel` or use the Kaggle API:  
> `kaggle datasets download -d biplobdey/puppy-or-bagel`

## Approach & Methodology

### Part 1 — Custom CNN from Scratch

```
Input (224×224×3)
→ Conv2D(32) + MaxPool
→ Conv2D(64) + MaxPool
→ Conv2D(128) + MaxPool
→ Conv2D(128) + MaxPool
→ Flatten
→ Dense(512, ReLU) + Dropout(0.5)
→ Dense(1, Sigmoid)
```

- **Data augmentation:** rotation, horizontal flip, zoom, width/height shift, shear
- **Callbacks:** EarlyStopping (patience=5), ModelCheckpoint (save best)
- **Result:** ~60–75% validation accuracy

### Part 2 — Transfer Learning with ResNet50

```
ResNet50 (ImageNet weights, frozen)
→ GlobalAveragePooling2D
→ Dense(256, ReLU) + Dropout(0.5)
→ Dense(1, Sigmoid)
```

- **Phase 1:** Froze entire ResNet50 base, trained only the custom head
- **Phase 2 (Fine-tuning):** Unfroze last 30 layers, recompiled with Adam (lr=0.0001)
- **Result:** **~92% validation accuracy**

### Performance Comparison

| Model | Val Accuracy | Training Time |
|-------|-------------|---------------|
| Custom CNN (scratch) | ~60–75% | Fast |
| ResNet50 (frozen head) | ~85–90% | Moderate |
| ResNet50 (fine-tuned) | **~92%** | Longer |

**Key insight:** A model pretrained on 1.2M ImageNet images already knows edges, textures, and object shapes. Fine-tuning those features for a new task is far more efficient than learning everything from random weights.

## Results & Evaluation

- Generated confusion matrices and classification reports for both models
- Visualized correct vs incorrect predictions with color-coded labels
- Confirmed that transfer learning outperforms a scratch CNN by ~17–32 percentage points on limited data

## Requirements

```
tensorflow>=2.12.0
numpy
matplotlib
seaborn
scikit-learn
Pillow
kaggle
```

Install with:
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn Pillow kaggle
```

## Files

| File | Description |
|------|-------------|
| `Module_03_Lab_Munachimso_Benson.ipynb` | Full notebook with both models, visualizations, and analysis |

## Learning Outcomes

- Understood how CNNs learn hierarchical visual features (edges → textures → objects)
- Experienced firsthand the power of transfer learning over training from scratch
- Learned how data augmentation reduces overfitting on small datasets
- Practiced fine-tuning pretrained models by selectively unfreezing layers

---

*ITAI-2376 · Spring 2026 · Houston Community College*

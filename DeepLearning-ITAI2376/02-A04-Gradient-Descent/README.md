# A04 — Gradient Descent: The Downhill Learning Adventure ⛰️

**Course:** ITAI-2376 Deep Learning | Spring 2026  
**Instructor:** Prof. Patricia McManus  
**Author:** Munachimso Benson  
**Type:** Creative Presentation

---

## Problem Statement

Gradient descent is one of the most important algorithms in all of deep learning, yet it's notoriously hard to explain intuitively. This assignment required presenting a core deep learning concept in a way that a complete beginner — specifically an 11-year-old — could grasp and remember.

## Approach & Methodology

The presentation used a **mountain hiking analogy** to explain gradient descent:

| Concept | Analogy |
|---------|---------|
| Loss function | Your height on a mountain |
| Goal | Reach the valley (minimize error) |
| Gradient | The slope — which direction is downhill? |
| Step | Moving one step down the slope |
| Learning rate | How big each step is |
| Convergence | Reaching flat ground (the minimum) |

### Key Concepts Covered

**The Core Loop:**
```
1. Make a prediction
2. Measure how wrong you are (loss)
3. Calculate the slope (gradient)
4. Take a step downhill
5. Repeat until you reach the bottom
```

**Learning Rate Effects:**
- Too large → you overshoot the valley and bounce around
- Too small → you move so slowly training takes forever
- Just right → smooth, steady descent to the minimum

**Types Introduced:**
- Batch gradient descent — uses all data each step
- Stochastic gradient descent (SGD) — one sample at a time
- Mini-batch — the balance used in most real training

## Results & Evaluation

- Delivered a story-driven presentation that made gradient descent visual and memorable
- Correctly connected the "Mystery Mountain" analogy to the real mathematical concepts of loss minimization and backpropagation
- Demonstrated understanding of learning rate sensitivity and its effect on convergence

## Learning Outcomes

- Understood gradient descent as the engine that drives all neural network training
- Learned how learning rate, batch size, and number of epochs interact during training
- Recognized gradient descent as the link between a model's predictions and its improvement
- Practiced translating mathematical optimization into human-scale intuition

## Files

| File | Description |
|------|-------------|
| `A04_Munachimso_Benson_ITAI2376.pptx` | Full presentation slides |

---

*ITAI-2376 · Spring 2026 · Houston Community College*

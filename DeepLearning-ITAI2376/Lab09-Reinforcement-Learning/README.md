# Lab 09 — Reinforcement Learning: Teach an Agent to Balance a Pole 🕹️

**Course:** ITAI-2376 Deep Learning | Spring 2026  
**Instructor:** Prof. Patricia McManus  
**Author:** Munachimso Benson  
**Type:** Hands-on Lab Notebook

---

## Problem Statement

How does an AI learn to make decisions without being given explicit rules? This lab introduced reinforcement learning through CartPole-v1 — a classic control problem where an agent must learn to balance a pole on a moving cart purely through trial and error, receiving a reward signal for every timestep it keeps the pole upright.

## Environment

**CartPole-v1** (OpenAI Gymnasium)

| Element | Description |
|---------|-------------|
| **State** | Cart position, cart velocity, pole angle, pole angular velocity |
| **Actions** | Push cart left (0) or right (1) |
| **Reward** | +1 for every timestep the pole stays upright |
| **Goal** | Survive for 500 timesteps (perfect score) |
| **Episode end** | Pole angle > ±12°, cart position > ±2.4 |

## Approach & Methodology

### Part 1 — Random Agent (Baseline)

A random agent that picks left or right with equal probability — no learning, no strategy.

**Result:** Average score ~21.2 points (range: 10–35)  
This established the "no learning" baseline every trained agent must beat.

### Part 2 — Q-Learning Agent

Q-Learning learns a **Q-table** — a lookup table mapping every (state, action) pair to an expected reward.

```
Q(s, a) ← Q(s, a) + α [r + γ · max Q(s', a') - Q(s, a)]
```

Where:
- `α` = learning rate (0.1)
- `γ` = discount factor (0.99)  
- `ε` = exploration rate (1.0 → 0.01 via decay)

**State discretization:** CartPole's 4 continuous state variables were bucketed into 10 bins each, creating a 10×10×10×10×2 Q-table.

**Epsilon-greedy exploration:**
```
With probability ε  → pick random action (explore)
With probability 1-ε → pick best known action (exploit)
```

### Exploration Rate Experiment

Compared three agents with different epsilon decay rates over 500 episodes:

| Agent | ε Decay | Last-50 Avg Score |
|-------|---------|-------------------|
| Fast decay | 0.990 | **~106.8** |
| Original | 0.995 | ~36.5 |
| Slow decay | 0.999 | ~38.1 |

**Finding:** Faster decay (committing to exploitation earlier) won within a 500-episode budget. With more episodes, slower decay would likely win by exploring more thoroughly.

### Part 3 — RLHF Connection

Connected CartPole's reward loop to Reinforcement Learning from Human Feedback (RLHF) — the technique used to align GPT-4 and Claude:

| CartPole | RLHF |
|----------|------|
| +1 per timestep | Human thumbs up/down |
| Pole angle = failure signal | Harmful/unhelpful response = penalty |
| Q-table = policy | Language model weights = policy |
| Epsilon-greedy | Temperature / sampling strategy |

**Final project connection:** Designed a reward model for a research summarization agent — reward = user satisfaction + factual accuracy. Identified reward hacking risk: confident-sounding responses could score high even if factually wrong.

## Results & Evaluation

| Agent | Avg Score | vs Random |
|-------|----------|-----------|
| Random baseline | 21.2 | — |
| Q-Learning (best) | **~106.8** | +5x improvement |

The Q-Learning agent began consistently outperforming the random baseline around episode 300–350.

## Requirements

```
gymnasium>=0.26.0
numpy
matplotlib
```

Install with:
```bash
pip install gymnasium numpy matplotlib
```

## Files

| File | Description |
|------|-------------|
| `L09_Munachimso_Benson_ITAI_2376.ipynb` | Full notebook with random agent, Q-Learning implementation, exploration experiment, and RLHF reflection |

## Learning Outcomes

- Understood the core RL framework: agent, environment, state, action, reward, policy
- Implemented Q-Learning from scratch including the Bellman update equation
- Learned how the explore/exploit tradeoff affects learning speed and final performance
- Connected tabular RL concepts to modern RLHF used in large language model alignment
- Recognized the limits of Q-tables for high-dimensional problems (like LLM state spaces)

---

*ITAI-2376 · Spring 2026 · Houston Community College*

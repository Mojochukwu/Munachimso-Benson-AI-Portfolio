# Midterm — Blog-to-Podcast Agent Blueprint 📋

**Course:** ITAI-2376 Deep Learning | Spring 2026  
**Instructor:** Prof. Patricia McManus  
**Author:** Munachimso Benson  
**Type:** Design Document / Agent Blueprint

---

## Problem Statement

Millions of people cannot access written blog content due to visual impairment, dyslexia, concentration difficulties, or simple time constraints. At the same time, creators miss out on reaching audio-first audiences. This blueprint designs an AI agent that solves both problems: take any blog URL and transform it into a human-sounding podcast episode automatically.

> **Note:** This is the design document (Midterm). The working implementation is in [`/Final-Blog-to-Podcast-Agent`](../Final-Blog-to-Podcast-Agent).

## Proposed Architecture

### Single Agent, Three Tools

| Tool | Purpose | Why This Tool |
|------|---------|---------------|
| BeautifulSoup | Scrape and clean blog HTML | Strips tags and noise, delivers clean paragraph text |
| GPT-4o-mini | Rewrite blog as spoken script | Understands full context — knows "click the link below" is meaningless in audio |
| OpenAI TTS | Convert script to audio `.mp3` | Neural model understands rhythm and cadence — sounds human, not robotic |

### Pipeline

```
User provides blog URL
        ↓
BeautifulSoup scrapes and cleans HTML
        ↓
GPT-4o-mini rewrites for audio format
        ↓
OpenAI TTS synthesises speech
        ↓
User receives .mp3 podcast episode
```

### Agent Framework — LangChain

Chose **LangChain** over AutoGen and CrewAI:

| Framework | Why Not Chosen |
|-----------|---------------|
| AutoGen | Designed for multi-agent coordination — overkill for one agent |
| CrewAI | Purpose-built for role-based multi-agent systems — unnecessary complexity |
| **LangChain** | ✅ Lightweight, clean tool registration, direct OpenAI integration, best documentation |

## Deep Learning Connections

| Module | Connection |
|--------|-----------|
| Module 10 (Agentic Systems) | GPT-4o-mini is the reasoning brain — not just copying text, actively understanding and restructuring content |
| Module 4 (Sequence Modeling + NLP) | OpenAI TTS is a neural seq2seq model — trained to understand rhythm, cadence, and prosody |
| Module 3 (Transformers) | GPT-4o-mini uses self-attention to understand context across the entire blog, not sentence-by-sentence |
| Module 2 (RNNs) | Sequence intuition applies — the rewriter processes paragraphs while maintaining flow and context |

## Anticipated Challenges & Mitigations

| Challenge | Mitigation |
|-----------|-----------|
| Scraping blocked sites (Medium, Forbes) | Manual text paste fallback so pipeline can still run |
| GPT-4o-mini leaving visual artifacts ("click the link") | Explicit system prompt instructions to strip all visual/link references |
| Long posts hitting TTS token limit | Word count check — summarize before sending if over limit |

## Estimated Cost

~$0.075 per average blog post (GPT-4o-mini + TTS combined)

## Files

| File | Description |
|------|-------------|
| `Blog-To-Podcast_Agent_Blueprint.pdf` | Full design document with architecture, framework analysis, and challenge mitigation |

## Learning Outcomes

- Practiced designing an AI system before building it — requirements, tools, trade-offs
- Learned to evaluate agent frameworks (LangChain vs AutoGen vs CrewAI) based on project needs
- Connected deep learning modules (Transformers, NLP, Sequence Models) to real product design
- Identified concrete failure modes and mitigation strategies before a single line of code was written

---

*ITAI-2376 · Spring 2026 · Houston Community College*

# Final Project — Blog-to-Podcast Agent (Working Implementation) 🤖🎙️

**Course:** ITAI-2376 Deep Learning | Spring 2026  
**Instructor:** Prof. Patricia McManus  
**Authors:** Munachimso Benson, Melvis Maduagwu, Christen Robinson, Kevin Moreira  
**Type:** Working AI Agent — Python / Jupyter Notebook  
**GitHub:** [github.com/Mojochukwu](https://github.com/Mojochukwu)

---

## Problem Statement

Written blog content is inaccessible to millions of people with visual impairment, dyslexia, or busy lifestyles. This project delivers a fully working AI agent that accepts any blog URL (or pasted text), rewrites it into a natural spoken script, and produces a human-sounding `.mp3` podcast episode — end-to-end, automatically.

This is the working implementation of the [Midterm Blueprint](../Midterm-Blog-to-Podcast-Blueprint).

## Demo

```
User: Convert this blog to a podcast: https://www.typeface.ai/blog/example

Agent: Scraping blog content...
       ✅ Scraped 26,037 characters
       Rewriting for audio...
       ✅ Script ready (3,622 chars | ~565 words | ~4 min listen time)
       Converting to audio...
       ✅ Audio saved to: podcast_output.mp3
```

## Architecture

### ReAct Agent Pattern

The agent follows the **ReAct loop** (Reason → Act → Observe → Respond):

```
User input
    ↓
[Reason]  Agent decides which tool to call
    ↓
[Act]     Tool executes (scrape or TTS)
    ↓
[Observe] Agent reads tool output
    ↓
[Respond] Agent returns result or calls next tool
```

### Tools

**Tool 1: `scrape_blog`**
```python
# Input:  Blog URL (string)
# Output: Clean plain text (HTML stripped)
# Tech:   requests + BeautifulSoup, extracts all <p> tags
# Fallback: Returns SCRAPE_FAILED message if site blocks request
```

**Tool 2: `convert_to_audio`**
```python
# Input:  Script text (string)
# Output: .mp3 audio file
# Tech:   OpenAI TTS (model: tts-1, voice: alloy)
# Smart:  _summarize_to_fit() compresses long scripts to stay under 4,096 char limit
#         using GPT-4o-mini — intelligent compression, not hard truncation
```

### Memory

```python
from langchain.memory import ConversationBufferMemory
# Persists conversation across turns
# User can say "make it more casual" and the agent recalls the prior script
```

## Full Pipeline

```
Blog URL / Pasted Text
        ↓
scrape_blog (BeautifulSoup)
        ↓
GPT-4o-mini rewrites for audio
(removes visual references, bullet points, hyperlink language)
        ↓
_summarize_to_fit() if > 4,096 chars
        ↓
Human approval checkpoint (optional)
        ↓
convert_to_audio (OpenAI TTS · voice: alloy)
        ↓
podcast_output.mp3
```

## Real Test Results

| Run | Blog Size | Compressed | GPT Cost | TTS Cost | Total |
|-----|-----------|-----------|----------|----------|-------|
| 1 (Typeface blog) | 26,037 chars | 3,622 chars | $0.002 | $0.027 | **$0.029** |
| 2 | 18,000 chars | 3,100 chars | $0.002 | $0.025 | **$0.027** |
| 3 | 15,000 chars | 2,800 chars | $0.001 | $0.022 | **$0.023** |

Average cost per blog post: **~$0.026**

## Requirements

```
langchain==0.2.16
langchain-openai
openai
requests
beautifulsoup4
python-dotenv
```

Install with:
```bash
pip install langchain==0.2.16 langchain-openai openai requests beautifulsoup4 python-dotenv
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/Mojochukwu/Munachimso-Benson-AI-Portfolio.git
cd Munachimso-Benson-AI-Portfolio/DeepLearning-ITAI2376/Final-Blog-to-Podcast-Agent
```

2. Set your OpenAI API key:
```bash
# Create a .env file
echo "OPENAI_API_KEY=your_key_here" > .env
```

3. Open and run the notebook:
```bash
jupyter notebook Munachimso_Benson_Melvis_Maduagwu_ITAI2376.ipynb
```

## Files

| File | Description |
|------|-------------|
| `Munachimso_Benson_Melvis_Maduagwu_ITAI2376.ipynb` | Full working agent notebook |
| `podcast_output.mp3` | Sample generated audio (if included) |

## Known Issues & Planned Improvements

| Issue | Status | Fix |
|-------|--------|-----|
| Some sites block scraping | Mitigated | Text paste fallback available |
| TTS 4,096 char limit | Solved | `_summarize_to_fit()` intelligently compresses |
| `stream_to_file` deprecation warning | Open | Migrate to `.with_streaming_response` |
| Fixed voice (alloy) | Planned | Add voice selection as user input |
| 4,096 char ceiling | Planned | Chunk-and-stitch for unlimited post length |

## Learning Outcomes

- Built a complete production-ready AI agent using the LangChain ReAct framework
- Implemented custom tools and integrated them with a language model reasoning loop
- Used `ConversationBufferMemory` to enable multi-turn context-aware interactions
- Solved real engineering constraints (TTS token limits) with an intelligent summarization fallback
- Understood the full cost model of a GPT-4o-mini + OpenAI TTS pipeline at ~$0.03/run

---

*ITAI-2376 · Spring 2026 · Houston Community College*

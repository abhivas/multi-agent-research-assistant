# 🔬 Multi-Agent Research Assistant

An AI-powered research assistant that uses **3 autonomous agents** orchestrated via **LangGraph** to research any topic, analyze sources, and generate comprehensive reports — all in one click.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://abhistic26-multi-agent-research-assistant.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────┐
│   LangGraph State Machine   │
│   (Orchestrator)            │
└──────┬──────┬──────┬────────┘
       │      │      │
       ▼      ▼      ▼
   ┌──────┐┌──────┐┌──────┐
   │Agent1││Agent2││Agent3│
   │Search││Analyz││Writer│
   └──┬───┘└──┬───┘└──┬───┘
      │       │       │
      ▼       ▼       ▼
   Tavily   FAISS   Gemini
   Search   Vector  LLM API
            Store
       │      │      │
       └──────┴──────┘
              │
              ▼
     Structured Research
     Report + Citations
```

## 🤖 How The 3 Agents Work

| Agent | Role | Tools Used |
|-------|------|-----------|
| 🔍 **Researcher** | Searches web, expands queries using LLM, collects 6-12 diverse sources | Tavily API / DuckDuckGo fallback |
| 📊 **Analyzer** | Reads each source, extracts key findings, scores relevance, builds FAISS vector store | Gemini LLM + FAISS + Sentence-BERT |
| ✍️ **Writer** | Synthesizes all analyzed data into a structured 800-1200 word report with citations | Gemini LLM |

## 🛠️ Tech Stack

- **Agent Orchestration:** LangGraph (state machine-based multi-agent workflow)
- **LLM:** Google Gemini 2.0 Flash (via LangChain)
- **Vector Store:** FAISS with Sentence-BERT (`all-MiniLM-L6-v2`) embeddings
- **Web Search:** Tavily API (with DuckDuckGo fallback)
- **Frontend:** Streamlit
- **Deployment:** Streamlit Cloud

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/Abhistic26/multi-agent-research-assistant.git
cd multi-agent-research-assistant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get API Keys
- **Google Gemini (required, free):** [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- **Tavily (optional, free tier):** [tavily.com](https://tavily.com)

### 4. Run the app
```bash
streamlit run app.py
```

## 📊 Sample Output

The system generates reports with:
- **Executive Summary** — 3-4 sentence overview
- **Key Findings** — organized by theme with `[Source N]` citations
- **Analysis** — synthesis of patterns and contradictions
- **Implications** — what this means for the field
- **References** — complete source list with URLs

## 🔑 Key Features

- **Multi-Agent Orchestration** — 3 specialized agents with defined roles, orchestrated via LangGraph state machine
- **Intelligent Query Expansion** — Researcher agent uses LLM to generate additional search queries for broader coverage
- **Source Quality Scoring** — Analyzer assigns relevance scores (0-1) and categorizes sources (statistics, theory, case study, etc.)
- **Vector Store Integration** — All analyzed content is embedded into FAISS for semantic retrieval
- **Citation-Aware Reports** — Writer generates reports with proper `[Source N]` citations
- **Downloadable Reports** — Export as Markdown
- **Graceful Fallbacks** — Works without Tavily API (uses web search fallback)

## 📁 Project Structure

```
multi-agent-research-assistant/
├── app.py                 # Main Streamlit application + all 3 agents
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .streamlit/
│   └── config.toml       # Streamlit theme configuration
└── assets/
    └── screenshot.png    # App screenshot (optional)
```

## 🧪 Example Research Topics

- `Impact of Large Language Models on Education`
- `AI agents in healthcare diagnostics`
- `RAG vs fine-tuning for enterprise LLMs`
- `Climate change impact on Indian agriculture`
- `Future of remote work in 2026`

## 📝 License

MIT License — feel free to use, modify, and distribute.

## 👤 Author

**Abhishek Singh**
- GitHub: [@Abhistic26](https://github.com/Abhistic26)
- LinkedIn: [abhishek-singh](https://www.linkedin.com/in/abhishek-singh-701405215/)
- Email: abhiabhishek2615@gmail.com

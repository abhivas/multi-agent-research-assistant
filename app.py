import streamlit as st
import os
import json
import time
from datetime import datetime
from typing import TypedDict, List, Annotated
from operator import add

# LangGraph & LangChain
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# Vector Store
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Web Search
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

import requests
from bs4 import BeautifulSoup

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide"
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1E3A5F 0%, #4A90D9 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .main-header p { color: #B8D4F0; margin: 0.3rem 0 0; font-size: 0.95rem; }
    .agent-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f8f9fa;
    }
    .agent-card h4 { margin: 0 0 0.3rem; }
    .source-card {
        border-left: 3px solid #4A90D9;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        background: #f0f7ff;
        border-radius: 0 6px 6px 0;
        font-size: 0.85rem;
    }
    .report-section {
        background: #ffffff;
        border: 1px solid #e8e8e8;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-running { background: #FFF3CD; color: #856404; }
    .badge-done { background: #D4EDDA; color: #155724; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Header
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🔬 Multi-Agent Research Assistant</h1>
    <p>Powered by LangGraph + Gemini + FAISS | Three AI agents collaborate to research any topic</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Sidebar - API Keys
# ============================================================
with st.sidebar:
    st.header("⚙️ Configuration")

    gemini_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        value=os.environ.get("GOOGLE_API_KEY", ""),
        help="Get your free key at aistudio.google.com/apikey"
    )

    tavily_key = st.text_input(
        "Tavily API Key (optional)",
        type="password",
        value=os.environ.get("TAVILY_API_KEY", ""),
        help="For better web search. Get free key at tavily.com"
    )

    st.divider()
    st.header("📖 How It Works")
    st.markdown("""
    **3 AI Agents work together:**

    1. 🔍 **Researcher** — Searches the web for relevant articles and papers

    2. 📊 **Analyzer** — Reads sources, extracts key information, stores in vector DB

    3. ✍️ **Writer** — Generates a structured research report with citations

    **Tech Stack:**
    - LangGraph (Agent Orchestration)
    - Google Gemini (LLM)
    - FAISS (Vector Store)
    - Sentence-BERT (Embeddings)
    - Tavily/Web (Search)
    """)

    st.divider()
    st.markdown("**Built by [Abhishek Singh](https://github.com/Abhistic26)**")


# ============================================================
# State Definition
# ============================================================
class ResearchState(TypedDict):
    query: str
    search_results: List[dict]
    analyzed_data: List[dict]
    vector_store_ready: bool
    report: str
    sources: List[str]
    status: str
    agent_logs: Annotated[List[str], add]


# ============================================================
# Helper: Web Search (Fallback without Tavily)
# ============================================================
def search_web_fallback(query: str, num_results: int = 5) -> List[dict]:
    """Simple web search using DuckDuckGo HTML scraping as fallback."""
    results = []
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Research Assistant Bot)"}
        resp = requests.get(
            f"https://html.duckduckgo.com/html/?q={query}",
            headers=headers, timeout=10
        )
        soup = BeautifulSoup(resp.text, "html.parser")
        for r in soup.select(".result")[:num_results]:
            title_el = r.select_one(".result__title")
            snippet_el = r.select_one(".result__snippet")
            link_el = r.select_one(".result__url")
            if title_el and snippet_el:
                results.append({
                    "title": title_el.get_text(strip=True),
                    "content": snippet_el.get_text(strip=True),
                    "url": link_el.get_text(strip=True) if link_el else "N/A"
                })
    except Exception as e:
        results.append({
            "title": "Search fallback info",
            "content": f"Web search encountered an issue: {str(e)}. Using LLM knowledge instead.",
            "url": "N/A"
        })
    return results


# ============================================================
# Agent 1: Researcher
# ============================================================
def researcher_agent(state: ResearchState) -> ResearchState:
    """Agent 1: Searches the web for relevant sources."""
    query = state["query"]
    logs = [f"🔍 **Researcher Agent** started | Query: '{query}'"]

    search_results = []

    # Try Tavily first
    if tavily_key and TAVILY_AVAILABLE:
        try:
            client = TavilyClient(api_key=tavily_key)
            response = client.search(query=query, max_results=6, search_depth="basic")
            for r in response.get("results", []):
                search_results.append({
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "url": r.get("url", "")
                })
            logs.append(f"✅ Tavily search returned {len(search_results)} results")
        except Exception as e:
            logs.append(f"⚠️ Tavily failed: {str(e)}, falling back to web search")

    # Fallback to DuckDuckGo
    if not search_results:
        search_results = search_web_fallback(query, num_results=6)
        logs.append(f"✅ Web search returned {len(search_results)} results")

    # Generate additional search queries using LLM
    if gemini_key:
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=gemini_key,
                temperature=0.3
            )
            expand_prompt = f"""Given the research topic: "{query}"
Generate 2 more specific search queries that would help find deeper information.
Return ONLY the queries, one per line. No numbering, no extra text."""

            response = llm.invoke([HumanMessage(content=expand_prompt)])
            extra_queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]

            for eq in extra_queries[:2]:
                extra_results = search_web_fallback(eq, num_results=3)
                search_results.extend(extra_results)
                logs.append(f"🔄 Expanded search: '{eq}' → {len(extra_results)} results")
        except Exception as e:
            logs.append(f"⚠️ Query expansion failed: {str(e)}")

    logs.append(f"📦 Total sources collected: {len(search_results)}")

    return {
        **state,
        "search_results": search_results,
        "status": "research_complete",
        "agent_logs": logs
    }


# ============================================================
# Agent 2: Analyzer
# ============================================================
def analyzer_agent(state: ResearchState) -> ResearchState:
    """Agent 2: Analyzes sources, extracts key info, builds vector store."""
    search_results = state["search_results"]
    logs = [f"📊 **Analyzer Agent** started | Processing {len(search_results)} sources"]

    if not gemini_key:
        logs.append("❌ No Gemini API key provided")
        return {**state, "analyzed_data": [], "agent_logs": logs}

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=gemini_key,
        temperature=0.2
    )

    analyzed_data = []

    # Process each source
    for i, source in enumerate(search_results):
        try:
            analyze_prompt = f"""Analyze this research source and extract key information.

Title: {source.get('title', 'N/A')}
Content: {source.get('content', 'N/A')[:2000]}
URL: {source.get('url', 'N/A')}

Extract the following in JSON format:
{{
    "key_findings": ["finding1", "finding2"],
    "relevance_score": 0.0 to 1.0,
    "category": "one of: statistics, theory, case_study, opinion, methodology",
    "summary": "2-3 sentence summary"
}}

Return ONLY valid JSON, no markdown backticks."""

            response = llm.invoke([HumanMessage(content=analyze_prompt)])
            raw = response.content.strip()

            # Clean JSON
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            analysis = json.loads(raw)
            analysis["source_title"] = source.get("title", "")
            analysis["source_url"] = source.get("url", "")
            analysis["original_content"] = source.get("content", "")
            analyzed_data.append(analysis)

        except (json.JSONDecodeError, Exception) as e:
            analyzed_data.append({
                "key_findings": [source.get("content", "")[:200]],
                "relevance_score": 0.5,
                "category": "general",
                "summary": source.get("content", "")[:300],
                "source_title": source.get("title", ""),
                "source_url": source.get("url", ""),
                "original_content": source.get("content", "")
            })

    logs.append(f"✅ Analyzed {len(analyzed_data)} sources")

    # Build FAISS vector store
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        texts = [d.get("summary", "") or d.get("original_content", "") for d in analyzed_data]
        texts = [t for t in texts if t.strip()]

        if texts:
            embeddings = model.encode(texts, show_progress_bar=False)
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(embeddings).astype('float32'))
            logs.append(f"✅ FAISS vector store built | {len(texts)} documents | {dimension}D embeddings")
            vector_ready = True
        else:
            logs.append("⚠️ No texts to embed")
            vector_ready = False
    except Exception as e:
        logs.append(f"⚠️ Vector store error: {str(e)}")
        vector_ready = False

    # Sort by relevance
    analyzed_data.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

    return {
        **state,
        "analyzed_data": analyzed_data,
        "vector_store_ready": vector_ready,
        "status": "analysis_complete",
        "agent_logs": logs
    }


# ============================================================
# Agent 3: Writer
# ============================================================
def writer_agent(state: ResearchState) -> ResearchState:
    """Agent 3: Generates structured research report."""
    analyzed_data = state["analyzed_data"]
    query = state["query"]
    logs = [f"✍️ **Writer Agent** started | Synthesizing {len(analyzed_data)} analyzed sources"]

    if not gemini_key:
        logs.append("❌ No Gemini API key")
        return {**state, "report": "Error: No API key", "agent_logs": logs}

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=gemini_key,
        temperature=0.4,
        max_output_tokens=4000
    )

    # Prepare source summaries for the writer
    source_context = ""
    sources_list = []
    for i, data in enumerate(analyzed_data[:10]):
        source_context += f"""
--- Source {i+1} ---
Title: {data.get('source_title', 'N/A')}
URL: {data.get('source_url', 'N/A')}
Category: {data.get('category', 'N/A')}
Relevance: {data.get('relevance_score', 'N/A')}
Key Findings: {', '.join(data.get('key_findings', []))}
Summary: {data.get('summary', 'N/A')}
"""
        sources_list.append(f"{data.get('source_title', 'N/A')} — {data.get('source_url', 'N/A')}")

    write_prompt = f"""You are a research report writer. Generate a comprehensive, well-structured research report on the following topic.

RESEARCH TOPIC: {query}

ANALYZED SOURCES:
{source_context}

REPORT REQUIREMENTS:
1. Start with an Executive Summary (3-4 sentences)
2. Introduction — why this topic matters
3. Key Findings — organized by theme, with citations like [Source 1], [Source 2]
4. Analysis — your synthesis of the findings, patterns, and contradictions
5. Implications — what this means for the field/industry
6. Conclusion — key takeaways
7. References — list all sources used

FORMATTING:
- Use markdown headers (##, ###)
- Use bullet points for key findings
- Include [Source N] citations throughout
- Be specific with data points and statistics when available
- Length: 800-1200 words
- Professional academic tone

Generate the report now:"""

    try:
        response = llm.invoke([HumanMessage(content=write_prompt)])
        report = response.content
        logs.append(f"✅ Report generated | {len(report.split())} words")
    except Exception as e:
        report = f"Error generating report: {str(e)}"
        logs.append(f"❌ Report generation failed: {str(e)}")

    return {
        **state,
        "report": report,
        "sources": sources_list,
        "status": "complete",
        "agent_logs": logs
    }


# ============================================================
# Build LangGraph
# ============================================================
def build_research_graph():
    """Build the multi-agent research workflow using LangGraph."""
    workflow = StateGraph(ResearchState)

    # Add nodes
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("analyzer", analyzer_agent)
    workflow.add_node("writer", writer_agent)

    # Define edges (sequential flow)
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "analyzer")
    workflow.add_edge("analyzer", "writer")
    workflow.add_edge("writer", END)

    return workflow.compile()


# ============================================================
# Main UI
# ============================================================
col1, col2 = st.columns([2, 1])

with col1:
    query = st.text_input(
        "🔎 Enter your research topic",
        placeholder="e.g., Impact of Large Language Models on Education in 2025",
        help="Be specific for better results"
    )

with col2:
    st.write("")
    st.write("")
    run_button = st.button("🚀 Start Research", type="primary", use_container_width=True)

# Example topics
st.markdown("**Try these:** `AI agents in healthcare` · `Climate change impact on agriculture in India` · `RAG vs fine-tuning for LLMs` · `Future of remote work 2026`")

st.divider()

# ============================================================
# Run Research
# ============================================================
if run_button and query:
    if not gemini_key:
        st.error("⚠️ Please enter your Gemini API key in the sidebar")
        st.stop()

    # Set API key
    os.environ["GOOGLE_API_KEY"] = gemini_key
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key

    # Build graph
    graph = build_research_graph()

    # Initial state
    initial_state = {
        "query": query,
        "search_results": [],
        "analyzed_data": [],
        "vector_store_ready": False,
        "report": "",
        "sources": [],
        "status": "starting",
        "agent_logs": []
    }

    # Progress tracking
    progress_bar = st.progress(0)
    status_container = st.container()
    agent_log_container = st.expander("📋 Agent Activity Log", expanded=True)

    # Run the graph
    with st.spinner("🔬 Research in progress..."):
        try:
            result = None
            step = 0

            for event in graph.stream(initial_state):
                step += 1

                for node_name, node_state in event.items():
                    # Update progress
                    if node_name == "researcher":
                        progress_bar.progress(33)
                        with status_container:
                            st.info("🔍 Agent 1: Researcher is searching the web...")
                    elif node_name == "analyzer":
                        progress_bar.progress(66)
                        with status_container:
                            st.info("📊 Agent 2: Analyzer is processing sources...")
                    elif node_name == "writer":
                        progress_bar.progress(100)
                        with status_container:
                            st.success("✍️ Agent 3: Writer has generated the report!")

                    # Show logs
                    if "agent_logs" in node_state:
                        with agent_log_container:
                            for log in node_state["agent_logs"]:
                                st.markdown(log)

                    result = node_state

            progress_bar.progress(100)

        except Exception as e:
            st.error(f"❌ Error during research: {str(e)}")
            st.stop()

    # ============================================================
    # Display Results
    # ============================================================
    if result and result.get("report"):
        st.divider()

        # Stats Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sources Found", len(result.get("search_results", [])))
        with col2:
            st.metric("Sources Analyzed", len(result.get("analyzed_data", [])))
        with col3:
            st.metric("Vector Store", "✅ Ready" if result.get("vector_store_ready") else "❌")
        with col4:
            st.metric("Report Words", len(result.get("report", "").split()))

        st.divider()

        # Report
        tab1, tab2, tab3 = st.tabs(["📄 Research Report", "📊 Source Analysis", "🔗 References"])

        with tab1:
            st.markdown(result["report"])

            # Download button
            st.download_button(
                label="📥 Download Report (Markdown)",
                data=result["report"],
                file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown"
            )

        with tab2:
            for i, data in enumerate(result.get("analyzed_data", [])):
                with st.expander(f"Source {i+1}: {data.get('source_title', 'N/A')[:80]}"):
                    cols = st.columns([1, 1, 1])
                    with cols[0]:
                        st.markdown(f"**Category:** {data.get('category', 'N/A')}")
                    with cols[1]:
                        score = data.get('relevance_score', 0)
                        st.markdown(f"**Relevance:** {'🟢' if score > 0.7 else '🟡' if score > 0.4 else '🔴'} {score:.2f}")
                    with cols[2]:
                        st.markdown(f"**URL:** [{data.get('source_url', 'N/A')[:40]}...]({data.get('source_url', '#')})")

                    st.markdown(f"**Summary:** {data.get('summary', 'N/A')}")

                    findings = data.get('key_findings', [])
                    if findings:
                        st.markdown("**Key Findings:**")
                        for f in findings:
                            st.markdown(f"- {f}")

        with tab3:
            st.markdown("### Sources Used")
            for i, source in enumerate(result.get("sources", []), 1):
                st.markdown(f"{i}. {source}")

elif run_button and not query:
    st.warning("Please enter a research topic first!")

# ============================================================
# Footer
# ============================================================
st.divider()
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.8rem;">
    Multi-Agent Research Assistant | Built with LangGraph + Gemini + FAISS + Streamlit<br>
    <a href="https://github.com/Abhistic26/multi-agent-research-assistant" target="_blank">GitHub</a> |
    By Abhishek Singh
</div>
""", unsafe_allow_html=True)

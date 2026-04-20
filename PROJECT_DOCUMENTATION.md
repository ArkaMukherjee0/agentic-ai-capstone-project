# Research Paper Q&A Agent — Project Documentation

**Name:** Arka Mukherjee
**Roll No:** 2328078
**Subject:** Agentic AI
**Instructor:** Dr. Kanthi Kiran Sirra, Sr. AI Engineer

---

## Problem Statement

PhD students and researchers frequently need to cross-reference multiple research papers to extract key findings, compare methodologies, and understand contributions. Manually reading and re-reading dense academic PDFs is time-consuming. There is no easy way to ask natural-language questions across a collection of papers and get grounded, faithful answers.

This project builds a conversational AI agent that:
- Answers questions about 12 landmark AI/ML papers using Retrieval-Augmented Generation (RAG)
- Maintains conversation memory across turns within a session
- Self-evaluates its answers for faithfulness and retries if below threshold
- Falls back to live ArXiv search when the user asks about papers outside the knowledge base

---

## Solution and Features

### 6 Mandatory Capabilities

| # | Capability | Implementation |
|---|-----------|----------------|
| 1 | **LangGraph StateGraph (3+ nodes)** | 8-node graph: memory → router → [retrieve / skip / tool] → answer → eval → save |
| 2 | **ChromaDB RAG (10+ documents)** | 12 PDFs parsed with `pypdf`, chunked to 150 words, embedded with `all-MiniLM-L6-v2`, stored in ChromaDB (~800–1200 chunks) |
| 3 | **Conversation memory** | `MemorySaver` checkpointer with `thread_id`; sliding window keeps last 6 messages (3 turns) |
| 4 | **Self-reflection** | `eval_node` scores faithfulness 0.0–1.0; retries `answer_node` if score < 0.7 (max 2 retries) |
| 5 | **Tool use** | ArXiv Search API — fetches live paper abstracts for out-of-KB queries; no API key required |
| 6 | **Deployment** | Streamlit UI (`capstone_streamlit.py`) deployed on Render; `render.yaml` included |

---

## Architecture

```
                        +----------------+
User Question --------> | memory_node    |  (append to history, sliding window)
                        +-------+--------+
                                |
                        +-------v--------+
                        | router_node    |  (LLM decides: retrieve / memory_only / tool)
                        +---+---+---+----+
                            |   |   |
               retrieve ----+   |   +---- tool
                            |   |
                       skip +   | memory_only
                            |   |
          +-----------------+   +----------+
          |                                |
  +-------v--------+            +----------v------+
  | retrieval_node |            | skip_retrieval  |
  | (ChromaDB top3)|            | (empty context) |
  +-------+--------+            +----------+------+
          |                                |
          +----------+  +------------------+
                     |  |
                +----v--v-------+       +----------+
                | tool_node     |       |          |
                | (ArXiv API)   +-----> | answer   |
                +---------------+       | _node    |
                                        | (LLM +   |
                                        | context) |
                                        +----+-----+
                                             |
                                        +----v-----+
                                        | eval_node|  (faithfulness score)
                                        +----+-----+
                                             |
                                   pass >= 0.7 or max retries
                                             |
                                        +----v-----+
                                        | save_node|  (append to history)
                                        +----+-----+
                                             |
                                            END
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | `llama-3.1-8b-instant` via `langchain-groq` |
| Graph Framework | LangGraph `StateGraph` + `MemorySaver` |
| Vector Database | ChromaDB (in-memory) |
| Embeddings | `all-MiniLM-L6-v2` (SentenceTransformers) |
| PDF Parsing | `pypdf` |
| UI | Streamlit |
| Deployment | Render (free tier) |
| Evaluation | RAGAS (`faithfulness`, `answer_relevancy`, `context_precision`) |
| External Tool | ArXiv API (no key required) |

---

## Knowledge Base

12 landmark AI/ML papers loaded from local ArXiv PDFs:

| ArXiv ID | Paper | Topic |
|----------|-------|-------|
| 1609.02907 | Graph Convolutional Networks (Kipf & Welling, 2017) | GNNs |
| 1706.03762 | Attention Is All You Need (Vaswani et al., 2017) | Transformers |
| 1810.04805 | BERT (Devlin et al., 2019) | Pre-training |
| 2005.11401 | RAG (Lewis et al., 2020) | Retrieval |
| 2005.14165 | GPT-3 (Brown et al., 2020) | Few-shot learning |
| 2006.11239 | DDPM (Ho et al., 2020) | Diffusion models |
| 2010.11929 | ViT (Dosovitskiy et al., 2021) | Vision Transformers |
| 2106.09685 | LoRA (Hu et al., 2022) | Fine-tuning |
| 2201.11903 | Chain-of-Thought (Wei et al., 2022) | Prompting |
| 2203.02155 | InstructGPT/RLHF (Ouyang et al., 2022) | Alignment |
| 2210.03629 | ReAct (Yao et al., 2023) | Agents |
| 2212.08073 | Constitutional AI (Bai et al., 2022) | Safety |

**Chunking:** 150 words per chunk, 20-word overlap. Typical KB size: ~800–1200 chunks.

---

## Tool: ArXiv Search

When the router decides `route = "tool"` (i.e., the user asks about a paper not likely in the KB), `tool_node` calls the free ArXiv API:

```
GET http://export.arxiv.org/api/query?search_query=all:{query}&max_results=3
```

Returns up to 3 paper titles, abstracts, and URLs as a string. No API key required. Fails gracefully with an error string on timeout.

---

## Test Results

12 test questions were run including 2 red-team tests:

| # | Question | Route | Status |
|---|----------|-------|--------|
| 1 | Main innovation in Attention Is All You Need? | retrieve | PASS |
| 2 | How does BERT differ from GPT in pre-training? | retrieve | PASS |
| 3 | What problem does RAG solve? | retrieve | PASS |
| 4 | Explain LoRA and parameter reduction | retrieve | PASS |
| 5 | What is the ReAct framework? | retrieve | PASS |
| 6 | How do diffusion models generate images? | retrieve | PASS |
| 7 | What is chain-of-thought prompting? | retrieve | PASS |
| 8 | What scale did GPT-3 introduce? | retrieve | PASS |
| 9 | BERT masking technique? (memory test) | memory_only | PASS |
| 10 | Find recent Constitutional AI papers on ArXiv | tool | PASS |
| 11 | Who won the 2024 Cricket World Cup? (red-team) | retrieve | PASS (declines) |
| 12 | GPT-3 trained on 1 billion tokens? (false premise) | retrieve | PASS (corrects) |

---

## RAGAS Baseline Scores

Run `day13_capstone.ipynb` Part 6 to populate:

| Metric | Score |
|--------|-------|
| Faithfulness | _run notebook_ |
| Answer Relevance | _run notebook_ |
| Context Precision | _run notebook_ |

---

## Deployment on Render

1. Push project to a public GitHub repo (include `papers/` directory)
2. Go to [render.com](https://render.com) → New → Web Service → Connect repo
3. Render auto-detects `render.yaml`
4. Add environment variable: `GROQ_API_KEY = <your key>`
5. Deploy — the app is live at the provided Render URL

---

## Unique Points

- **PDF-native KB**: Instead of hand-written summaries, the agent reads directly from ArXiv PDFs using `pypdf`, giving it access to full paper text including methods, equations context, and ablations.
- **Two-stage retrieval fallback**: If the KB doesn't have an answer (`memory_only` route), the ArXiv tool fetches live abstracts — the agent never says "I can't help" for a genuine research question.
- **Self-healing eval loop**: The faithfulness gate catches hallucinations before the answer is saved to conversation memory, so degraded answers don't poison future turns.

---

## Future Improvements

1. **Hybrid BM25 + dense retrieval**: Replace pure vector search with LangChain's `EnsembleRetriever` (BM25 + ChromaDB) to improve context precision for exact technical terms like "LoRA rank" or "constitutional AI critique-revision".
2. **PDF upload at runtime**: Add a Streamlit file uploader with `pypdf` so researchers can query their own papers, not just the 12 pre-loaded ones.
3. **Citation-aware chunking**: Chunk at section boundaries (Abstract, Introduction, Methods, Results) instead of fixed word counts, so retrieved chunks align with paper structure and are more coherent.

"""
agent.py — Research Paper Q&A Agent (shared module)

Usage:
    from agent import build_agent
    app, embedder, collection = build_agent()
    result = app.invoke({"question": "..."}, config={"configurable": {"thread_id": "t1"}})
"""
import os
import pypdf
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import TypedDict, List

import chromadb
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from sentence_transformers import SentenceTransformer

load_dotenv()

PAPER_METADATA = {
    "1609.02907": "Graph Convolutional Networks (Kipf & Welling, 2017)",
    "1706.03762": "Attention Is All You Need (Vaswani et al., 2017)",
    "1810.04805": "BERT: Bidirectional Transformers (Devlin et al., 2019)",
    "2005.11401": "Retrieval-Augmented Generation (Lewis et al., 2020)",
    "2005.14165": "GPT-3: Few-Shot Learners (Brown et al., 2020)",
    "2006.11239": "Denoising Diffusion Probabilistic Models (Ho et al., 2020)",
    "2010.11929": "Vision Transformer / ViT (Dosovitskiy et al., 2021)",
    "2106.09685": "LoRA: Low-Rank Adaptation (Hu et al., 2022)",
    "2201.11903": "Chain-of-Thought Prompting (Wei et al., 2022)",
    "2203.02155": "InstructGPT / RLHF (Ouyang et al., 2022)",
    "2210.03629": "ReAct: Reasoning and Acting (Yao et al., 2023)",
    "2212.08073": "Constitutional AI (Bai et al., 2022)",
}

PAPERS_DIR             = "papers"
CHUNK_WORDS            = 150
OVERLAP_WORDS          = 20
FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES       = 2


class CapstoneState(TypedDict):
    question:     str
    messages:     List[dict]
    route:        str
    retrieved:    str
    sources:      List[str]
    tool_result:  str
    answer:       str
    faithfulness: float
    eval_retries: int
    paper_topics: List[str]


def _load_pdf_chunks(pdf_path: str) -> list[str]:
    reader    = pypdf.PdfReader(pdf_path)
    full_text = " ".join(page.extract_text() or "" for page in reader.pages)
    words     = full_text.split()
    step      = CHUNK_WORDS - OVERLAP_WORDS
    return [
        " ".join(words[i : i + CHUNK_WORDS])
        for i in range(0, len(words), step)
        if len(" ".join(words[i : i + CHUNK_WORDS]).strip()) > 50
    ]


def _build_kb(embedder: SentenceTransformer):
    client = chromadb.Client()
    try:
        client.delete_collection("capstone_kb")
    except Exception:
        pass
    collection = client.create_collection("capstone_kb")

    docs = []
    for fname in sorted(os.listdir(PAPERS_DIR)):
        if not fname.endswith(".pdf"):
            continue
        arxiv_id = fname.split("v")[0]
        title    = PAPER_METADATA.get(arxiv_id, arxiv_id)
        for idx, chunk in enumerate(_load_pdf_chunks(os.path.join(PAPERS_DIR, fname))):
            docs.append({"id": f"{arxiv_id}_{idx:04d}", "topic": title, "text": chunk})

    texts = [d["text"] for d in docs]
    collection.add(
        documents=texts,
        embeddings=embedder.encode(texts).tolist(),
        ids=[d["id"] for d in docs],
        metadatas=[{"topic": d["topic"]} for d in docs],
    )
    return collection


def arxiv_search(query: str, max_results: int = 3) -> str:
    url = (
        "http://export.arxiv.org/api/query"
        f"?search_query=all:{urllib.parse.quote(query)}"
        f"&max_results={max_results}&sortBy=relevance"
    )
    try:
        resp    = urllib.request.urlopen(url, timeout=10)
        root    = ET.parse(resp).getroot()
        ns      = {"a": "http://www.w3.org/2005/Atom"}
        entries = root.findall("a:entry", ns)
        out     = []
        for e in entries:
            title   = e.find("a:title",   ns).text.strip().replace("\n", " ")
            summary = e.find("a:summary", ns).text.strip().replace("\n", " ")[:400]
            link    = e.find("a:id",      ns).text.strip()
            out.append(f"Title: {title}\nAbstract: {summary}\nURL: {link}")
        return "\n\n".join(out) or "No ArXiv results found."
    except Exception as ex:
        return f"ArXiv search unavailable: {ex}"


def build_agent():
    llm      = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    collection = _build_kb(embedder)

    def memory_node(state: CapstoneState) -> dict:
        msgs = state.get("messages", []) + [{"role": "user", "content": state["question"]}]
        return {"messages": msgs[-6:]}

    def router_node(state: CapstoneState) -> dict:
        question = state["question"]
        messages = state.get("messages", [])
        recent   = "; ".join(f"{m['role']}: {m['content'][:60]}" for m in messages[-3:-1]) or "none"
        prompt   = (
            "You are a router for a Research Paper Q&A assistant covering 12 AI/ML papers.\n"
            "Options:\n"
            "- retrieve: search local knowledge base for paper content\n"
            "- memory_only: answer from conversation history only\n"
            "- tool: use ArXiv Search for papers not in knowledge base\n"
            f"Recent: {recent}\nQuestion: {question}\n"
            "Reply with ONLY one word: retrieve / memory_only / tool"
        )
        decision = llm.invoke(prompt).content.strip().lower()
        if "memory" in decision:  decision = "memory_only"
        elif "tool" in decision:  decision = "tool"
        else:                     decision = "retrieve"
        return {"route": decision}

    def retrieval_node(state: CapstoneState) -> dict:
        q_emb   = embedder.encode([state["question"]]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=3)
        chunks  = results["documents"][0]
        topics  = [m["topic"] for m in results["metadatas"][0]]
        context = "\n\n---\n\n".join(f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks)))
        return {
            "retrieved":    context,
            "sources":      topics,
            "paper_topics": list(dict.fromkeys(topics)),
        }

    def skip_retrieval_node(state: CapstoneState) -> dict:
        return {"retrieved": "", "sources": [], "paper_topics": []}

    def tool_node(state: CapstoneState) -> dict:
        return {"tool_result": arxiv_search(state["question"])}

    def answer_node(state: CapstoneState) -> dict:
        question     = state["question"]
        retrieved    = state.get("retrieved", "")
        tool_result  = state.get("tool_result", "")
        messages     = state.get("messages", [])
        eval_retries = state.get("eval_retries", 0)

        parts = []
        if retrieved:   parts.append(f"KNOWLEDGE BASE:\n{retrieved}")
        if tool_result: parts.append(f"ARXIV SEARCH:\n{tool_result}")
        context = "\n\n".join(parts)

        sys_prompt = (
            "You are a Research Paper Q&A assistant. "
            "Answer using ONLY the context below. "
            "If the answer is not in the context, say: 'I don't have that information.'\n\n"
            + (context or "No context available.")
        )
        if eval_retries > 0:
            sys_prompt += (
                "\n\nPrevious answer flagged for low faithfulness. "
                "Be more precise and ground every claim in the context."
            )

        lc_msgs = [SystemMessage(content=sys_prompt)]
        for m in messages[:-1]:
            lc_msgs.append(
                HumanMessage(content=m["content"]) if m["role"] == "user"
                else AIMessage(content=m["content"])
            )
        lc_msgs.append(HumanMessage(content=question))
        return {"answer": llm.invoke(lc_msgs).content}

    def eval_node(state: CapstoneState) -> dict:
        answer  = state.get("answer", "")
        context = state.get("retrieved", "")[:500]
        retries = state.get("eval_retries", 0)
        if not context:
            return {"faithfulness": 1.0, "eval_retries": retries + 1}
        prompt = (
            "Rate faithfulness 0.0-1.0. Reply with only a number.\n"
            f"Context: {context}\nAnswer: {answer[:300]}"
        )
        try:
            score = float(llm.invoke(prompt).content.strip().split()[0].replace(",", "."))
            score = max(0.0, min(1.0, score))
        except Exception:
            score = 0.5
        return {"faithfulness": score, "eval_retries": retries + 1}

    def save_node(state: CapstoneState) -> dict:
        msgs = state.get("messages", []) + [{"role": "assistant", "content": state["answer"]}]
        return {"messages": msgs}

    def route_decision(state: CapstoneState) -> str:
        r = state.get("route", "retrieve")
        if r == "tool":        return "tool"
        if r == "memory_only": return "skip"
        return "retrieve"

    def eval_decision(state: CapstoneState) -> str:
        score   = state.get("faithfulness", 1.0)
        retries = state.get("eval_retries", 0)
        if score >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES:
            return "save"
        return "answer"

    graph = StateGraph(CapstoneState)
    for name, fn in [
        ("memory",   memory_node),
        ("router",   router_node),
        ("retrieve", retrieval_node),
        ("skip",     skip_retrieval_node),
        ("tool",     tool_node),
        ("answer",   answer_node),
        ("eval",     eval_node),
        ("save",     save_node),
    ]:
        graph.add_node(name, fn)

    graph.set_entry_point("memory")
    graph.add_edge("memory", "router")
    graph.add_conditional_edges(
        "router", route_decision,
        {"retrieve": "retrieve", "skip": "skip", "tool": "tool"},
    )
    for src in ("retrieve", "skip", "tool"):
        graph.add_edge(src, "answer")
    graph.add_edge("answer", "eval")
    graph.add_conditional_edges(
        "eval", eval_decision, {"answer": "answer", "save": "save"}
    )
    graph.add_edge("save", END)

    app = graph.compile(checkpointer=MemorySaver())
    return app, embedder, collection

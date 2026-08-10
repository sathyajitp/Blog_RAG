# Blog RAG

A minimal Retrieval-Augmented Generation (RAG) system that answers questions
over 12 blog posts, with an LLM-as-judge evaluation pipeline
and a retrieval-only evaluation pipeline, both built on LangSmith.

Built as a hands-on walkthrough of the full RAG lifecycle - index, retrieve,
generate, evaluate - using free/open tooling end to end (no OpenAI key
required).

## What it does

1. **Scrapes** the blog posts and chunks them into passages.
2. **Embeds** the chunks locally (no API calls) and persists them to a small
   on-disk vector store.
3. **Re-indexes incrementally** - re-running `index.py` hashes each URL's
   content and only re-embeds posts that actually changed.
4. **Answers questions** about the posts through a retrieve-then-generate CLI.
5. **Evaluates generation** with 4 LLM-as-judge graders - correctness,
   relevance, groundedness, and retrieval relevance - logged to LangSmith.
6. **Evaluates retrieval in isolation** - hit rate@k and MRR against a
   labeled (question → expected source URL) set, with no LLM call, also
   logged to LangSmith.

## How it works

```
                   ┌──────────────┐
                   │  index.py    │  (run once, or when source docs change)
                   │              │
   12 blog URLs ──▶│ scrape (bs4) │
                   │ chunk (RCTS) │
                   │ embed (HF)   │
                   └──────┬───────┘
                          ▼
                   vectorstore.json  (persisted InMemoryVectorStore)
                          │
        ┌─────────────────┼──────────────────────┐
        ▼                 ▼                       ▼
  ┌───────────┐   ┌───────────────┐   ┌───────────────────────┐
  │  rag.py   │   │  evaluate.py  │   │ retrieval_metrics.py   │
  │           │   │               │   │                        │
  │ retrieve  │   │ retrieve      │   │ retrieve only          │
  │ generate  │   │ generate      │   │ (no LLM call)          │
  │ (Groq)    │   │ grade (Groq   │   │ hit_rate@k, MRR        │
  │ interact- │   │ as judge)     │   │ evaluators             │
  │ ive CLI   │   │               │   │                        │
  └───────────┘   └───────┬───────┘   └───────────┬────────────┘
                           └──────────┬────────────┘
                                      ▼
                            LangSmith dashboard
```

`rag_core.py` holds everything shared by `rag.py`, `evaluate.py`, and
`retrieval_metrics.py` - config, embeddings, LLM, retriever, and the
`rag_bot()` function - so the CLI and both evaluation harnesses are
guaranteed to run the *exact* same pipeline.

Stack: `HuggingFaceEmbeddings` (local, `all-MiniLM-L6-v2`) for embeddings,
`ChatGroq` (`llama-3.3-70b-versatile`) for generation and grading,
`InMemoryVectorStore` persisted to JSON for storage, LangSmith for eval
tracking.

## Setup

```bash
python -m venv venv
source venv/Scripts/activate     # Windows (Git Bash) — use venv\Scripts\activate in cmd/PowerShell
pip install -r requirements.txt

cp .env.example .env             # fill in your keys
python index.py                  # one-time indexing
python rag.py                    # ask questions
python evaluate.py               # run the generation eval (LLM-as-judge, LangSmith)
python retrieval_metrics.py      # run the retrieval-only eval (hit rate@k, MRR, LangSmith)
```

## What's next

This is deliberately a small, complete slice of a RAG system, not a finished
one. Planned next steps - chunking and embedding-model experiments, reranking, hybrid
search, and eventually swapping in a real vector DB for deployment.

## Files

| File | Purpose |
|---|---|
| `rag_core.py` | Shared config, embeddings, LLM, retriever, `rag_bot()` |
| `index.py` | One-time indexing: scrape → chunk → embed → persist to `vectorstore.json` |
| `rag.py` | Interactive CLI: load persisted store, ask questions |
| `evaluate.py` | Creates a LangSmith dataset, runs `rag_bot` over it, grades with 4 LLM-as-judge evaluators |
| `retrieval_metrics.py` | Retrieval-only eval: hit rate@k, MRR against a labeled question → source set, no LLM call, logged to LangSmith |
| `.env` / `.env.example` | API keys and config|
| `vectorstore.json` | Persisted embeddings|
| `index_manifest.json` | Per-URL content hash + chunk ids, used by `index.py` to skip re-embedding unchanged posts|

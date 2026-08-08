# Darius Foroux Blog RAG

A minimal Retrieval-Augmented Generation (RAG) system that answers questions
over 3 Darius Foroux blog posts, with an LLM-as-judge evaluation pipeline
built on LangSmith.

- [How to Build a Quiet and Boring Life That You Love](https://dariusforoux.com/how-to-build-a-quiet-and-boring-life-that-you-love/)
- [Your Side Project Will Never Launch (and How to Fix That)](https://dariusforoux.com/your-side-project-will-never-launch-and-how-to-fix-that/)
- [The Usefulness of Suffering](https://dariusforoux.com/the-usefulness-of-suffering/)

Built as a hands-on walkthrough of the full RAG lifecycle — index, retrieve,
generate, evaluate — using free/open tooling end to end (no OpenAI key
required). See [`Notes.md`](Notes.md) for the detailed rationale behind every
design choice.

## What it does

1. **Scrapes** the 3 blog posts and chunks them into ~20 passages.
2. **Embeds** the chunks locally (no API calls) and persists them to a small
   on-disk vector store.
3. **Answers questions** about the posts through a retrieve-then-generate CLI.
4. **Evaluates** itself with 4 LLM-as-judge graders — correctness, relevance,
   groundedness, and retrieval relevance — logged to LangSmith for
   inspection.

## How it works

```
                 ┌──────────────┐
                 │  index.py    │  (run once, or when source docs change)
                 │              │
  3 blog URLs ──▶│ scrape (bs4) │
                 │ chunk (RCTS) │
                 │ embed (HF)   │
                 └──────┬───────┘
                        ▼
                 vectorstore.json  (persisted InMemoryVectorStore)
                        │
           ┌────────────┴────────────┐
           ▼                         ▼
     ┌───────────┐            ┌──────────────┐
     │  rag.py   │            │ evaluate.py  │
     │           │            │              │
     │ retrieve  │            │ retrieve     │
     │ generate  │            │ generate     │
     │ (Groq)    │            │ grade (Groq  │
     │ interact- │            │ as judge) ── │──▶ LangSmith dashboard
     │ ive CLI   │            │ via LangSmith│
     └───────────┘            └──────────────┘
```

`rag_core.py` holds everything shared by `rag.py` and `evaluate.py` — config,
embeddings, LLM, retriever, and the `rag_bot()` function — so the CLI and the
evaluation harness are guaranteed to run the *exact* same pipeline.

Stack: `HuggingFaceEmbeddings` (local, `all-MiniLM-L6-v2`) for embeddings,
`ChatGroq` (`llama-3.3-70b-versatile`) for generation and grading,
`InMemoryVectorStore` persisted to JSON for storage, LangSmith for eval
tracking. Why each of these instead of the more common OpenAI/hosted-DB
defaults is explained in [`Notes.md`](Notes.md).

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt

cp .env.example .env             # fill in your keys
python index.py                  # one-time indexing
python rag.py                    # ask questions
python evaluate.py               # run the LangSmith evaluation
```

`.env` fields are documented in [`Notes.md`](Notes.md#setup).

## Results

The evaluation dataset currently has 3 Q&A pairs (one per blog post). Latest
run scored **1.0 (True) across all 4 evaluators** — correctness, relevance,
groundedness, and retrieval relevance — for every example.

That's a clean pass, but not a strong signal yet: with only 3 questions, all
answerable and all clearly covered by the corpus, there's little for the
graders to catch. The dataset hasn't yet included edge cases like
unanswerable questions, ambiguous phrasing, or multi-hop questions across
posts — see [`todo.md`](todo.md) for the plan to stress-test this properly.

## What's next

This is deliberately a small, complete slice of a RAG system, not a finished
one. Planned next steps — bigger eval sets, retrieval-only metrics (hit
rate/MRR), chunking and embedding-model experiments, reranking, hybrid
search, and eventually swapping in a real vector DB for deployment — are
tracked in [`todo.md`](todo.md).

## Files

| File | Purpose |
|---|---|
| `rag_core.py` | Shared config, embeddings, LLM, retriever, `rag_bot()` |
| `index.py` | One-time indexing: scrape → chunk → embed → persist to `vectorstore.json` |
| `rag.py` | Interactive CLI: load persisted store, ask questions |
| `evaluate.py` | Creates a LangSmith dataset, runs `rag_bot` over it, grades with 4 LLM-as-judge evaluators |
| `Notes.md` | Detailed architectural decisions and tradeoffs |
| `todo.md` | Planned experiments and improvements |
| `.env` / `.env.example` | API keys and config (gitignored; never commit `.env`) |
| `vectorstore.json` | Persisted embeddings (gitignored; regenerate with `index.py`) |

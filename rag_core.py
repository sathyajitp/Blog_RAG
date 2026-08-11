import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

load_dotenv()

URLS = [
    "https://dariusforoux.com/how-to-build-a-quiet-and-boring-life-that-you-love/",
    "https://dariusforoux.com/your-side-project-will-never-launch-and-how-to-fix-that/",
    "https://dariusforoux.com/the-usefulness-of-suffering/",
    "https://dariusforoux.com/why-your-personal-brand-is-your-most-valuable-asset-in-the-ai-era/",
    "https://dariusforoux.com/how-to-deal-with-negative-people-and-criticism/",
    "https://dariusforoux.com/the-courage-to-live-on-your-own-terms/",
    "https://dariusforoux.com/is-life-worse-today-compared-to-10-years-ago/",
    "https://dariusforoux.com/this-thinking-mistake-destroys-your-life/",
    "https://dariusforoux.com/how-to-express-yourself-clearly/",
    "https://dariusforoux.com/the-mental-model-that-changed-my-entire-life/",
    "https://dariusforoux.com/what-does-it-actually-mean-to-live-a-good-life/",
    "https://dariusforoux.com/why-modern-life-feels-so-hard-even-when-youre-doing-well/",
    
]

VECTORSTORE_PATH = Path(__file__).parent / "vectorstore.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"{name} is not set. Add it to your .env file (see .env.example)."
        )
    return value


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_llm() -> ChatGroq:
    require_env("GROQ_API_KEY")
    return ChatGroq(model=LLM_MODEL, temperature=0)


def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL)


def rerank(reranker: CrossEncoder, query: str, docs: list[Document], top_k: int) -> list[Document]:
    """Score each doc against the query with a cross-encoder and keep the top_k."""
    if not docs:
        return docs
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


def load_retriever(k: int = 4):
    """Load the persisted vector store from disk. Run index.py first to create it."""
    if not VECTORSTORE_PATH.exists():
        raise SystemExit(
            f"No vector store found at {VECTORSTORE_PATH}.\n"
            "Run `python index.py` once to index the source documents, then re-run this script."
        )
    vectorstore = InMemoryVectorStore.load(str(VECTORSTORE_PATH), embedding=get_embeddings())
    return vectorstore.as_retriever(search_kwargs={"k": k})


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


RAG_INSTRUCTIONS = """You are a helpful assistant who is good at analyzing source information and answering questions.
Use the following source documents to answer the user's question.
If you don't know the answer, just say that you don't know.
Keep the answer concise.

<context>
{context}
</context>"""


def make_rag_bot(retriever, llm, reranker: CrossEncoder | None = None, top_k: int = 4):
    def rag_bot(question: str) -> dict:
        docs = retriever.invoke(question)
        if reranker is not None:
            docs = rerank(reranker, question, docs, top_k)
        instructions = RAG_INSTRUCTIONS.format(context=format_docs(docs))
        ai_msg = llm.invoke(
            [
                {"role": "system", "content": instructions},
                {"role": "user", "content": question},
            ]
        )
        return {"answer": ai_msg.content, "documents": docs}

    return rag_bot

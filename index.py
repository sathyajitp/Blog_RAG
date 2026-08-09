import hashlib
import json

import bs4
import requests
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_core import URLS, VECTORSTORE_PATH, get_embeddings

# Tracks, per source URL, the content hash it was indexed at and the vectorstore
# chunk ids produced from it. Lets re-runs skip re-embedding unchanged pages.
MANIFEST_PATH = VECTORSTORE_PATH.with_name("index_manifest.json")


def load_web_page(url: str) -> list[Document]:
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    soup = bs4.BeautifulSoup(response.text, "html.parser")
    article = soup.find("article") or soup.find("main") or soup.body

    # The "Read Next" related-posts widget shows a randomized set of other
    # posts on every request, which would make the page hash non-deterministic
    # (breaking incremental re-indexing) and pollutes chunks with unrelated
    # post titles. Strip it before extracting text.
    related = article.find(class_="entry-related")
    if related is not None:
        related.decompose()

    return [Document(page_content=article.get_text(separator="\n"), metadata={"source": url})]


def hash_content(docs: list[Document]) -> str:
    """SHA-256 over a URL's raw scraped text, used to detect unchanged pages."""
    combined = "\n".join(doc.page_content for doc in docs)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest: dict) -> None:
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def load_vectorstore(embeddings) -> InMemoryVectorStore:
    """Reuse the persisted vector store if present, otherwise start empty."""
    if VECTORSTORE_PATH.exists():
        return InMemoryVectorStore.load(str(VECTORSTORE_PATH), embedding=embeddings)
    return InMemoryVectorStore(embedding=embeddings)


def main():
    embeddings = get_embeddings()
    vectorstore = load_vectorstore(embeddings)
    manifest = load_manifest()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    changed, unchanged = 0, 0

    for url in URLS:
        print(f"Checking {url}...")
        docs = load_web_page(url)
        content_hash = hash_content(docs)
        entry = manifest.get(url)

        # Incremental re-indexing: if the page's content hash matches what we
        # last indexed, its chunks are already in the vector store — skip
        # re-embedding it entirely.
        if entry and entry["hash"] == content_hash:
            print("  Unchanged, skipping re-embedding.")
            unchanged += 1
            continue

        # Content is new or changed: drop any previously indexed chunks for
        # this URL before adding the freshly split/embedded ones, so stale
        # chunks don't linger alongside the updated ones.
        if entry and entry.get("ids"):
            vectorstore.delete(ids=entry["ids"])

        doc_splits = text_splitter.split_documents(docs)
        chunk_ids = [f"{url}::{i}" for i in range(len(doc_splits))]
        vectorstore.add_documents(documents=doc_splits, ids=chunk_ids)

        manifest[url] = {"hash": content_hash, "ids": chunk_ids}
        changed += 1
        print(f"  Re-embedded {len(doc_splits)} chunks.")

    # Drop chunks for URLs that were removed from URLS since the last run,
    # so the vector store and manifest never drift from the current source list.
    removed_urls = set(manifest) - set(URLS)
    for url in removed_urls:
        vectorstore.delete(ids=manifest[url].get("ids", []))
        del manifest[url]

    vectorstore.dump(str(VECTORSTORE_PATH))
    save_manifest(manifest)
    print(
        f"Done. {changed} URL(s) re-embedded, {unchanged} unchanged, "
        f"{len(removed_urls)} removed. Saved vector store to {VECTORSTORE_PATH}"
    )


if __name__ == "__main__":
    main()

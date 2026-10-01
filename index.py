import hashlib # hashing data(sha-256)
import json # handling json data
import bs4 # beautiful soup: This is used for scraping HTML pages. Generates a parse tree
import requests # Sending HTTP requests
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter #Chunking method
from rag_core import URLS, VECTORSTORE_PATH, get_embeddings # import from rag_core

# Tracks, per source URL, the content hash it was indexed at and the vectorstore
# chunk ids produced from it. Lets re-runs skip re-embedding unchanged pages.
MANIFEST_PATH = VECTORSTORE_PATH.with_name("index_manifest.json")

# Loading a web page using the URLS
def load_web_page(url: str) -> list[Document]:
    # {"User-Agent": "Mozilla/5.0"} tells a web server that an HTTP request comes from a standard web browser, even if it is sent by a script or program. Done to avoid being blocked by websites due to bot assumption.
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    # raise_for_status() checks if the HTTP request went through. Throws an error if error.
    response.raise_for_status()
    # Parse the returned response into a parse-tree.
    soup = bs4.BeautifulSoup(response.text, "html.parser")
    # Finding tags with name article or main or body
    article = soup.find("article") or soup.find("main") or soup.body

    # The "Read Next" related-posts widget shows a randomized set of other
    # posts on every request, which would make the page hash non-deterministic
    # (breaking incremental re-indexing) and pollutes chunks with unrelated
    # post titles. Strip it before extracting text.
    related = article.find(class_="entry-related")
    if related is not None:
        related.decompose()

    # return a langchain document with text and metadata
    return [Document(page_content=article.get_text(separator="\n"), metadata={"source": url})]


# SHA-256 over a URL's raw scraped text, used to detect unchanged pages.
def hash_content(docs: list[Document]) -> str:
    combined = "\n".join(doc.page_content for doc in docs)

    # combined - the text you want to hash
    # .encode("utf-8") - converts the string into a bytes object used as input to hash function
    # hashlib.sha256(bytes) - passes encoded bytes into sha256 algorithm
    # Returns the final hash value as a readable hexadecimal string instead of raw binary bytes.
    # utf-8 is a format which converts human readable to machine readable text so that a computer can store, transmit and understand
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()

# Load the manifest json file
def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        # converts a json to a dictionary and returns it
        return json.load(f)

# Writing into the manifest file and saving it
def save_manifest(manifest: dict) -> None:
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        # Opposite of json.load(). Writes manifest json to json file f
        json.dump(manifest, f, indent=2)

# Factory Function to Load vectorstore
def load_vectorstore(embeddings) -> InMemoryVectorStore:
    # Reuse the persisted vector store if present, otherwise start empty.
    # Loads existing vectorstore from disk to RAM
    if VECTORSTORE_PATH.exists():
        return InMemoryVectorStore.load(str(VECTORSTORE_PATH), embedding=embeddings)
    return InMemoryVectorStore(embedding=embeddings)


def main():
    embeddings = get_embeddings() # Get embedding model
    vectorstore = load_vectorstore(embeddings) # load vectorstore
    manifest = load_manifest() # load manifest file
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # initiate chunker

    changed, unchanged = 0, 0
    # For each url present in the corpus, it is first loaded, scraped, parsed into parse tree, and its content and metadata(url source) is stored as a langchain document object.
    # Then the text content is hashed using sha256 and returned as a hexadecimal
    # The hash already stored for the url is retrieved.
    # Incremental reindexing check - if the hash of the url content scraped is already present in the manifest, which means that it has been scraped before, then not re-embedding it saves a lot of time and memory.
    # If the hash is not the same, means content has changed. Delete the url's chunk ids from vectorstore so that new ones can be added.
    # Then, the docs are chunked, chunk ids are created, embedded and stored in vectorstore.

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
        # .add_documents embeds the content and stores it in the vectorstore
        vectorstore.add_documents(documents=doc_splits, ids=chunk_ids)

        # manifest updated
        manifest[url] = {"hash": content_hash, "ids": chunk_ids}
        changed += 1
        print(f"  Re-embedded {len(doc_splits)} chunks.")

    # Drop chunks for URLs that were removed from URLS since the last run,
    # so the vector store and manifest never drift from the current source list.
    removed_urls = set(manifest) - set(URLS)
    for url in removed_urls:
        vectorstore.delete(ids=manifest[url].get("ids", []))
        del manifest[url]

    # Save vectorstore to disk so that its not lost(exists in RAM so its wiped out when closed)
    vectorstore.dump(str(VECTORSTORE_PATH))
    save_manifest(manifest)
    print(
        f"Done. {changed} URL(s) re-embedded, {unchanged} unchanged, "
        f"{len(removed_urls)} removed. Saved vector store to {VECTORSTORE_PATH}"
    )


if __name__ == "__main__":
    main()

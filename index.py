import bs4
import requests
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_core import URLS, VECTORSTORE_PATH, get_embeddings

def load_web_page(url: str) -> list[Document]:
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    soup = bs4.BeautifulSoup(response.text, "html.parser")
    article = soup.find("article") or soup.find("main") or soup.body
    return [Document(page_content=article.get_text(separator="\n"), metadata={"source": url})]


def main():
    print("Loading blog posts...")
    docs_list = [doc for url in URLS for doc in load_web_page(url)]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    doc_splits = text_splitter.split_documents(docs_list)
    print(f"Split into {len(doc_splits)} chunks.")

    print("Embedding chunks (local model, first run downloads the model)...")
    embeddings = get_embeddings()
    vectorstore = InMemoryVectorStore.from_documents(documents=doc_splits, embedding=embeddings)

    vectorstore.dump(str(VECTORSTORE_PATH))
    print(f"Saved vector store to {VECTORSTORE_PATH}")


if __name__ == "__main__":
    main()

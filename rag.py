from rag_core import get_llm, load_retriever, make_rag_bot

def main():
    retriever = load_retriever()
    llm = get_llm()
    rag_bot = make_rag_bot(retriever, llm)

    print("RAG system ready. Ask questions about the 3 Darius Foroux blog posts.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in {"exit", "quit", ""}:
            break
        result = rag_bot(question)
        print(f"\nAnswer: {result['answer']}\n")


if __name__ == "__main__":
    main()

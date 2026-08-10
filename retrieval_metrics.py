from langsmith import Client

from rag_core import load_retriever, require_env

require_env("LANGSMITH_API_KEY")

client = Client()

DATASET_NAME = "Darius Foroux Retrieval Eval"
K_VALUES = [1, 3, 5]

# One question per indexed blog post, labeled with the URL it should be
# retrieved from.
EXAMPLES = [
    {
        "inputs": {"question": "What does it mean to build a quiet and boring life?"},
        "outputs": {"expected_source": "https://dariusforoux.com/how-to-build-a-quiet-and-boring-life-that-you-love/"},
    },
    {
        "inputs": {"question": "Why do side projects never launch, and how do you fix that?"},
        "outputs": {"expected_source": "https://dariusforoux.com/your-side-project-will-never-launch-and-how-to-fix-that/"},
    },
    {
        "inputs": {"question": "How can suffering be useful?"},
        "outputs": {"expected_source": "https://dariusforoux.com/the-usefulness-of-suffering/"},
    },
    {
        "inputs": {"question": "Why is your personal brand valuable in the AI era?"},
        "outputs": {"expected_source": "https://dariusforoux.com/why-your-personal-brand-is-your-most-valuable-asset-in-the-ai-era/"},
    },
    {
        "inputs": {"question": "How should you deal with negative people and criticism?"},
        "outputs": {"expected_source": "https://dariusforoux.com/how-to-deal-with-negative-people-and-criticism/"},
    },
    {
        "inputs": {"question": "What does it take to have the courage to live on your own terms?"},
        "outputs": {"expected_source": "https://dariusforoux.com/the-courage-to-live-on-your-own-terms/"},
    },
    {
        "inputs": {"question": "Is life worse today compared to 10 years ago?"},
        "outputs": {"expected_source": "https://dariusforoux.com/is-life-worse-today-compared-to-10-years-ago/"},
    },
    {
        "inputs": {"question": "What thinking mistake destroys your life?"},
        "outputs": {"expected_source": "https://dariusforoux.com/this-thinking-mistake-destroys-your-life/"},
    },
    {
        "inputs": {"question": "How can you express yourself more clearly?"},
        "outputs": {"expected_source": "https://dariusforoux.com/how-to-express-yourself-clearly/"},
    },
    {
        "inputs": {"question": "What mental model changed the author's entire life?"},
        "outputs": {"expected_source": "https://dariusforoux.com/the-mental-model-that-changed-my-entire-life/"},
    },
    {
        "inputs": {"question": "What does it actually mean to live a good life?"},
        "outputs": {"expected_source": "https://dariusforoux.com/what-does-it-actually-mean-to-live-a-good-life/"},
    },
    {
        "inputs": {"question": "Why does modern life feel so hard even when you're doing well?"},
        "outputs": {"expected_source": "https://dariusforoux.com/why-modern-life-feels-so-hard-even-when-youre-doing-well/"},
    },
]


def get_or_create_dataset():
    if client.has_dataset(dataset_name=DATASET_NAME):
        return client.read_dataset(dataset_name=DATASET_NAME)
    dataset = client.create_dataset(dataset_name=DATASET_NAME)
    client.create_examples(dataset_id=dataset.id, examples=EXAMPLES)
    return dataset


def ranks_for_sources(sources: list[str], expected_source: str) -> list[int]:
    """1-indexed ranks, within the retrieved list, of chunks from the expected source."""
    return [i + 1 for i, source in enumerate(sources) if source == expected_source]


def make_hit_rate_evaluator(k: int):
    def hit_rate(outputs: dict, reference_outputs: dict) -> bool:
        ranks = ranks_for_sources(outputs["sources"], reference_outputs["expected_source"])
        return any(r <= k for r in ranks)

    hit_rate.__name__ = f"hit_rate@{k}"
    return hit_rate


def mrr(outputs: dict, reference_outputs: dict) -> float:
    ranks = ranks_for_sources(outputs["sources"], reference_outputs["expected_source"])
    return 1 / min(ranks) if ranks else 0.0


def main():
    dataset = get_or_create_dataset()
    retriever = load_retriever(k=max(K_VALUES))

    def target(inputs: dict) -> dict:
        docs = retriever.invoke(inputs["question"])
        return {"sources": [doc.metadata.get("source") for doc in docs]}

    evaluators = [make_hit_rate_evaluator(k) for k in K_VALUES] + [mrr]

    results = client.evaluate(
        target,
        data=dataset.name,
        evaluators=evaluators,
        experiment_prefix="darius-foroux-retrieval",
        metadata={"retriever_k": max(K_VALUES)},
    )
    print(results)


if __name__ == "__main__":
    main()

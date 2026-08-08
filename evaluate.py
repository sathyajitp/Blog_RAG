from typing_extensions import Annotated, TypedDict
from langsmith import Client
from rag_core import get_llm, load_retriever, make_rag_bot, require_env

require_env("LANGSMITH_API_KEY")

client = Client()

DATASET_NAME = "Darius Foroux Blogs Q&A"

EXAMPLES = [
    {
        "inputs": {"question": "What is the main idea behind building a quiet and boring life?"},
        "outputs": {
            "answer": "A quiet life doesn't mean a small or unfulfilling one — it means being "
            "undistracted and fully present, free from the constant pursuit of more and "
            "anxiety about the future."
        },
    },
    {
        "inputs": {"question": "Why do side projects often fail to launch, and how can you fix that?"},
        "outputs": {
            "answer": "Side projects often stall because of fear and perfectionism; the fix is "
            "to ship something small and imperfect instead of waiting until it feels ready."
        },
    },
    {
        "inputs": {"question": "How can suffering be useful, according to the blog post?"},
        "outputs": {
            "answer": "Suffering can be useful because facing hardship builds resilience and "
            "character, and teaches lessons that comfort alone cannot."
        },
    },
]


def get_or_create_dataset():
    if client.has_dataset(dataset_name=DATASET_NAME):
        return client.read_dataset(dataset_name=DATASET_NAME)
    dataset = client.create_dataset(dataset_name=DATASET_NAME)
    client.create_examples(dataset_id=dataset.id, examples=EXAMPLES)
    return dataset


# --- Evaluators (mirrors the LangSmith RAG tutorial) ---------------------


class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]


CORRECTNESS_INSTRUCTIONS = """You are a teacher grading a quiz. You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer.
(2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""


class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "Provide the score on whether the answer addresses the question"]


RELEVANCE_INSTRUCTIONS = """You are a teacher grading a quiz. You will be given a QUESTION and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""


class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[bool, ..., "Provide the score on if the answer hallucinates from the documents"]


GROUNDED_INSTRUCTIONS = """You are a teacher grading a quiz. You will be given FACTS and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS.
(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""


class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool, ..., "True if the retrieved documents are relevant to the question, False otherwise"
    ]


RETRIEVAL_RELEVANCE_INSTRUCTIONS = """You are a teacher grading a quiz. You will be given a QUESTION and a set of FACTS provided by the student. Here is the grade criteria to follow:
(1) Your goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

Relevance:
A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
A relevance value of False means that the FACTS are completely unrelated to the QUESTION.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""


def build_evaluators():
    grader_llm = get_llm()

    correctness_llm = grader_llm.with_structured_output(CorrectnessGrade)
    relevance_llm = grader_llm.with_structured_output(RelevanceGrade)
    grounded_llm = grader_llm.with_structured_output(GroundedGrade)
    retrieval_relevance_llm = grader_llm.with_structured_output(RetrievalRelevanceGrade)

    def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
        answers = (
            f"QUESTION: {inputs['question']}\n"
            f"GROUND TRUTH ANSWER: {reference_outputs['answer']}\n"
            f"STUDENT ANSWER: {outputs['answer']}"
        )
        grade = correctness_llm.invoke(
            [
                {"role": "system", "content": CORRECTNESS_INSTRUCTIONS},
                {"role": "user", "content": answers},
            ]
        )
        return grade["correct"]

    def relevance(inputs: dict, outputs: dict) -> bool:
        answer = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs['answer']}"
        grade = relevance_llm.invoke(
            [
                {"role": "system", "content": RELEVANCE_INSTRUCTIONS},
                {"role": "user", "content": answer},
            ]
        )
        return grade["relevant"]

    def groundedness(inputs: dict, outputs: dict) -> bool:
        doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
        answer = f"FACTS: {doc_string}\nSTUDENT ANSWER: {outputs['answer']}"
        grade = grounded_llm.invoke(
            [
                {"role": "system", "content": GROUNDED_INSTRUCTIONS},
                {"role": "user", "content": answer},
            ]
        )
        return grade["grounded"]

    def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
        doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
        answer = f"FACTS: {doc_string}\nQUESTION: {inputs['question']}"
        grade = retrieval_relevance_llm.invoke(
            [
                {"role": "system", "content": RETRIEVAL_RELEVANCE_INSTRUCTIONS},
                {"role": "user", "content": answer},
            ]
        )
        return grade["relevant"]

    return [correctness, relevance, groundedness, retrieval_relevance]


def main():
    dataset = get_or_create_dataset()

    retriever = load_retriever()
    llm = get_llm()
    rag_bot = make_rag_bot(retriever, llm)

    def target(inputs: dict) -> dict:
        return rag_bot(inputs["question"])

    evaluators = build_evaluators()

    results = client.evaluate(
        target,
        data=dataset.name,
        evaluators=evaluators,
        experiment_prefix="darius-foroux-rag",
        metadata={"llm": "llama-3.3-70b-versatile"},
    )
    print(results)


if __name__ == "__main__":
    main()

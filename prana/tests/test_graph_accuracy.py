from graph.chat_graph import build_chat_graph


# =========================================================
# FINAL 25-QUESTION LANGGRAPH TEST
# =========================================================

TEST_CASES = [

    # ---------------------------------------------------------
    # 1–3: GREETINGS
    # ---------------------------------------------------------

    {
        "question": "Hi",
        "expected": "greeting",
    },
    {
        "question": "Hello, how are you?",
        "expected": "greeting",
    },
    {
        "question": "Hey there!",
        "expected": "greeting",
    },


    # ---------------------------------------------------------
    # 4–11: DIRECT KNOWLEDGE QUESTIONS
    # ---------------------------------------------------------

    {
        "question": "Who is responsible for backend development?",
        "expected": "Anjam",
    },
    {
        "question": "Who is responsible for frontend development?",
        "expected": "Karthik",
    },
    {
        "question": "Who is responsible for FastAPI development?",
        "expected": "Siddappa",
    },
    {
        "question": "What embedding model is used for dense embeddings?",
        "expected": "BAAI/bge-small-en-v1.5",
    },
    {
        "question": "What embedding system is used for sparse retrieval?",
        "expected": "Qdrant/bm25",
    },
    {
        "question": "Which vector database does the project use?",
        "expected": "Qdrant",
    },
    {
        "question": "What does the Express backend handle?",
        "expected": "authentication",
    },
    {
        "question": "What does the FastAPI service provide?",
        "expected": "RAG",
    },


    # ---------------------------------------------------------
    # 12–17: MEDIUM / INDIRECT QUESTIONS
    # ---------------------------------------------------------

    {
        "question": "Who would be responsible for implementing the React interface?",
        "expected": "Karthik",
    },
    {
        "question": "Which team member works mainly on the user-facing part of the application?",
        "expected": "Karthik",
    },
    {
        "question": "Which person handles the service that connects the RAG system to the application?",
        "expected": "Siddappa",
    },
    {
        "question": "How does information travel from the frontend to the RAG system?",
        "expected": "FastAPI",
    },
    {
        "question": "How does the system combine semantic and keyword-based retrieval?",
        "expected": "hybrid retrieval",
    },
    {
        "question": "What happens to documents before they are stored in the knowledge base?",
        "expected": "Qdrant",
    },


    # ---------------------------------------------------------
    # 18–21: NEGATIVE / CONTRASTIVE QUESTIONS
    # ---------------------------------------------------------

    {
        "question": "Does Anjam handle frontend development?",
        "expected": "No",
    },
    {
        "question": "Does Karthik handle FastAPI development?",
        "expected": "No",
    },
    {
        "question": "Does Siddappa handle authentication?",
        "expected": "No",
    },
    {
        "question": "Does Anjam work with React-based frontend components?",
        "expected": "No",
    },


    # ---------------------------------------------------------
    # 22–25: UNSUPPORTED QUESTIONS
    # ---------------------------------------------------------

    {
        "question": "What database is used to store user passwords?",
        "expected": "I don't have that information",
    },
    {
        "question": "What programming language is used to implement authentication?",
        "expected": "I don't have that information",
    },
    {
        "question": "What is the monthly cloud infrastructure cost of the project?",
        "expected": "I don't have that information",
    },
    {
        "question": "Who is the project manager?",
        "expected": "I don't have that information",
    },
]


# =========================================================
# BUILD LANGGRAPH
# =========================================================

graph = build_chat_graph()


# =========================================================
# SIMPLE ANSWER CHECK
# =========================================================

def answer_matches(actual_answer: str, expected: str) -> bool:
    """
    Simple benchmark check.

    We are intentionally keeping this lightweight.
    The complete LangGraph is still responsible for producing
    the answer.
    """

    if not actual_answer:
        return False

    actual = actual_answer.lower().strip()
    expected = expected.lower().strip()

    # Greeting
    if expected == "greeting":
        greeting_words = [
            "hello",
            "hi",
            "hey",
            "how can i help",
            "how may i help",
        ]

        return any(word in actual for word in greeting_words)

    # Unsupported answer
    if expected == "i don't have that information":
        unsupported_phrases = [
            "i don't have that information",
            "don't have that information",
            "not available in the knowledge base",
            "not in the knowledge base",
            "information is not available",
        ]

        return any(phrase in actual for phrase in unsupported_phrases)

    # Negative answer
    if expected == "no":
        return (
            actual.startswith("no")
            or " no." in actual
            or " no," in actual
            or "does not" in actual
            or "is not responsible" in actual
            or "not responsible" in actual
        )

    # Normal knowledge answer
    return expected in actual


# =========================================================
# RUN ALL QUESTIONS THROUGH LANGGRAPH
# =========================================================

correct = 0

print("\n" + "=" * 90)
print("FINAL 25-QUESTION LANGGRAPH END-TO-END TEST")
print("=" * 90)

for i, test in enumerate(TEST_CASES, start=1):

    question = test["question"]
    expected = test["expected"]

    print("\n" + "-" * 90)
    print(f"QUESTION {i}")
    print(f"Q:        {question}")
    print(f"EXPECTED: {expected}")

    try:

        # -------------------------------------------------
        # ACTUAL PRODUCTION FLOW
        #
        # Question
        #    ↓
        # LangGraph
        #    ↓
        # Retrieval
        #    ↓
        # Qdrant
        #    ↓
        # Dense + Sparse
        #    ↓
        # Hybrid
        #    ↓
        # Reranker
        #    ↓
        # Gemini
        #    ↓
        # Final answer
        # -------------------------------------------------

        result = graph.invoke({
            "message": question
        })

        actual_answer = result.get("answer", "")

        # Get retrieval information for debugging only
        candidates = result.get("retrieved_candidates", [])

        top_score = None

        if candidates:
            top_score = candidates[0].rerank_score

        actual_sufficient = result.get(
            "evidence_sufficient",
            None,
        )

        # -------------------------------------------------
        # CHECK FINAL ANSWER
        # -------------------------------------------------

        is_correct = answer_matches(
            actual_answer,
            expected,
        )

        print(f"ACTUAL ANSWER: {actual_answer}")
        print(f"TOP RERANK SCORE: {top_score}")
        print(f"EVIDENCE SUFFICIENT: {actual_sufficient}")

        if is_correct:
            print("RESULT: ✅ CORRECT")
            correct += 1
        else:
            print("RESULT: ❌ WRONG")

    except Exception as exc:

        print(f"RESULT: ❌ ERROR")
        print(f"ERROR: {exc}")


# =========================================================
# FINAL ACCURACY
# =========================================================

total = len(TEST_CASES)

accuracy = (
    correct / total
) * 100


print("\n" + "=" * 90)
print("FINAL RESULT")
print("=" * 90)

print(f"Correct : {correct}/{total}")
print(f"Accuracy: {accuracy:.2f}%")

print("=" * 90)

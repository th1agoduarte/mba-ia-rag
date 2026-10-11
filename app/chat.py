from langchain.messages import HumanMessage

from app.llm import get_chat_model


def main() -> None:
    model = get_chat_model()

    print("No-RAG chat started.")
    print("This chat does not use the internal knowledge base yet.")
    print("Type 'exit' to quit.")

    while True:
        question = input("\n> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        for chunk in model.stream([HumanMessage(question)]):
            print(chunk.text, end="", flush=True)
        print()


if __name__ == "__main__":
    main()

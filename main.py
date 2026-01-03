from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel

# fetch openai api key from env file
import dotenv
dotenv.load_dotenv()


my_agent = Agent(
    'openai:gpt-5',
    instructions=(
        "You are a friendly CLI tutor helping the user learn pydantic-ai. "
        "Keep answers short (1-3 sentences)."
    ),
    # tools=[greet_tool],
)

@my_agent.tool_plain
async def greet_tool(mood: str) -> str:
    """A simple tool that returns a personalized greeting."""
    print(f"[greet_tool] called with mood: {mood}")
    return f"[{mood.title()} greeting from tool] Nice to meet you!"

async def main() -> None:
    """Run a tiny REPL so you can play with pydantic-ai."""

    import asyncio

    print("=== pydantic-ai demo ===")
    print("Type 'exit' or Ctrl+C to quit.\n")

    while True:
        try:
            user_msg = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_msg:
            continue
        if user_msg.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        result = await my_agent.run(user_msg)
        print("Bot:", result.output, "\n", "usage:", result.usage(), "\n")
        print("All messages:", result.all_messages(), "\n")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

## pydantic-ai test app

This is a tiny console demo to help you start learning **pydantic-ai**.

It creates a simple agent backed by `OpenAIChatModel` with one tool (`greet_tool`) and runs a REPL loop in `main.py`.

### Prerequisites

- Python 3.10+
- An OpenAI-compatible API key exported as `OPENAI_API_KEY`.

```bash
export OPENAI_API_KEY="sk-..."
```

### Install dependencies

From the project root:

```bash
pip install -e .
```

or, if you just want to install the dependencies defined in `pyproject.toml`:

```bash
pip install .
```

### Run the demo

```bash
python -m dso-ai-test.main
```

You should see:

```text
=== pydantic-ai demo ===
Type 'exit' or Ctrl+C to quit.
```

Type a message (for example `say hi with a happy mood using the greet_tool`) and the agent will respond using the model and tool.

### Next steps

- Add more tools in `main.py` to call external APIs or run calculations.
- Change the system prompt to adjust the agent’s personality.
- Introduce Pydantic models for structured outputs instead of plain strings.

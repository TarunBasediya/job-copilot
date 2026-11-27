from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import TodoListMiddleware

from inspect_utils import show_tool_calls, show_todo_history
from llm import get_model
from research_tools import web_search, fetch_page

model = get_model()

OUTPUT_DIR = Path("agent_output")
OUTPUT_DIR.mkdir(exist_ok=True)
backend = FilesystemBackend(root_dir=OUTPUT_DIR.resolve(), virtual_mode=True)

SYSTEM_PROMPT = """You are a company research assistant helping a job seeker.

Rules:
1. Begin with write_todos. Mark an item completed only after its work is done.
2. Use web_search first, then fetch_page on at most 2 promising results.
3. Only state facts that appear in tool results. If something could not be
   found, write "Not found" for it instead of guessing.
4. Save the final brief to /company_research.md with these sections:
   Overview, Products, Recent News, Sources.
5. Under Sources, list every URL you actually used.
6. Finish with a 3-line summary.
"""

agent = create_deep_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[web_search, fetch_page],
    middleware=[TodoListMiddleware()],
    backend=backend,
)

company = "Anthropic"
result = agent.invoke(
    {"messages": [{"role": "user", "content": f"Research the company: {company}"}]}
)

show_tool_calls(result)
show_todo_history(result)

print("\n=== company_research.md ===")
brief = OUTPUT_DIR / "company_research.md"
print(brief.read_text(encoding="utf-8") if brief.exists() else "(file not created)")

print("\n=== FINAL ANSWER ===")
print(result["messages"][-1].content)
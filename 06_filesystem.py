from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import TodoListMiddleware
from llm import get_model

from inspect_utils import show_tool_calls, show_todo_history

model = get_model()

# Agent ka apna sandboxed folder: yahi uska "root" (/) hoga
OUTPUT_DIR = Path("agent_output")
OUTPUT_DIR.mkdir(exist_ok=True)

backend = FilesystemBackend(root_dir=OUTPUT_DIR.resolve(), virtual_mode=True)

SYSTEM_PROMPT = """You are a career assistant that helps analyse job descriptions.

Workflow rules:
1. ALWAYS begin by calling write_todos with a checklist that covers every
   deliverable the user asked for.
2. Keep exactly one item in_progress at a time. Mark an item completed ONLY
   after the work for it is truly done. A verification item may be marked
   completed only AFTER you have read both files back with read_file.
3. Save files in the root directory with exactly these paths:
   /skills.md and /interview.md. Do not create any folders.
4. After writing the files, verify that every must-have skill in /skills.md is
   covered by at least one question in /interview.md. Fix gaps with edit_file.
5. Finish with a short summary.
"""

agent = create_deep_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    middleware=[TodoListMiddleware()],
    backend=backend,
)

jd_text = Path("sample_jd.txt").read_text(encoding="utf-8")

task = f"""
Here is a job description:

{jd_text}

Please:
1. Extract the required skills and split them into must-have and nice-to-have.
2. Save the result in /skills.md.
3. Write 5 interview questions in /interview.md. Together, the questions must
   cover every must-have skill.
4. Verify that coverage and finish with a short summary.
"""

result = agent.invoke({"messages": [{"role": "user", "content": task}]})


def show_first_todo_call(result):
    """Pehli write_todos call ke raw args aur tool ka jawab dikhata hai."""
    msgs = result["messages"]
    for i, m in enumerate(msgs):
        for tc in getattr(m, "tool_calls", None) or []:
            if tc["name"] != "write_todos":
                continue
            print("\n=== FIRST write_todos: RAW ARGS ===")
            print(tc["args"])
            for nxt in msgs[i + 1:]:
                if getattr(nxt, "tool_call_id", None) == tc["id"]:
                    print("tool response:", str(nxt.content)[:300])
                    break
            return


show_tool_calls(result)
show_todo_history(result)
show_first_todo_call(result)

print("\n=== STATE KEYS ===")
print(list(result.keys()))
print("files in state:", list((result.get("files") or {}).keys()))

print("\n=== FILES ON REAL DISK ===")
for p in sorted(OUTPUT_DIR.rglob("*")):
    if p.is_file():
        print(p)

print("\n=== FINAL ANSWER ===")
print(result["messages"][-1].content)


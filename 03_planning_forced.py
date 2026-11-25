from pathlib import Path

from deepagents import create_deep_agent
from llm import get_model

from inspect_utils import show_tool_calls, show_todo_history, show_state

model = get_model()

SYSTEM_PROMPT = """You are a career assistant that helps analyse job descriptions.

Workflow rules:
1. ALWAYS begin by calling write_todos with a checklist that covers every
   deliverable the user asked for.
2. Keep exactly one item in_progress at a time. Mark an item completed
   immediately after finishing it.
3. After writing the files, read them back with read_file and verify that
   every must-have skill in skills.md is covered by at least one question in
   interview.md. If something is missing, fix it with edit_file.
4. Finish with a short summary.
"""

agent = create_deep_agent(model=model, system_prompt=SYSTEM_PROMPT)

jd_text = Path("sample_jd.txt").read_text(encoding="utf-8")

task = f"""
Here is a job description:

{jd_text}

Please:
1. Extract the required skills and split them into must-have and nice-to-have.
2. Save the result in skills.md.
3. Write 5 interview questions in interview.md. Together, the questions must
   cover every must-have skill.
4. Verify that coverage and finish with a short summary.
"""

result = agent.invoke({"messages": [{"role": "user", "content": task}]})

show_tool_calls(result)
show_todo_history(result)
show_state(result)
print("\n=== FINAL ANSWER ===")
print(result["messages"][-1].content)
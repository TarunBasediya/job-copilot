from pathlib import Path

from deepagents import create_deep_agent
from llm import get_model

model = get_model()

agent = create_deep_agent(
    model=model,
    system_prompt="You are a career assistant that helps analyse job descriptions."
)

jd_text = Path("sample_jd.txt").read_text(encoding="utf-8")

task = f"""
Here is a job description:

{jd_text}

Please do ALL of the following:
1. Extract the required skills.
2. Split them into must-have and nice-to-have.
3. Save the result in a file called skills.md.
4. Write 5 interview questions for this role in a file called interview.md.
5. Finish with a short summary of what you did.
"""

result = agent.invoke({"messages": [{"role": "user", "content": task}]})

def file_text(value):
    # Version ke hisaab se file ka value string ya dict ho sakta hai, dono handle karo
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, list):
            return "\n".join(content)
        if isinstance(content, str):
            return content
    return str(value)

# (a) Saare tool calls order mein
print("=== (a) TOOL CALLS IN ORDER ===")
for m in result["messages"]:
    for tc in getattr(m, "tool_calls", None) or []:
        print("-", tc["name"])

# (b) Har write_todos call ne kya checklist bheji
print("\n=== (b) EVERY write_todos CALL ===")
call_no = 0
for m in result["messages"]:
    for tc in getattr(m, "tool_calls", None) or []:
        if tc["name"] == "write_todos":
            call_no += 1
            print(f"\nwrite_todos call #{call_no}:")
            for t in tc["args"].get("todos", []):
                print(f"   [{t.get('status')}] {t.get('content')}")


# (c) Final todos state
print("\n=== (c) FINAL TODOS IN STATE ===")
print(result.get("todos"))

# (d) Files
print("\n=== (d) FILES ===")
for path, value in (result.get("files") or {}).items():
    print(f"\n{path}")
    print(file_text(value)[:300])


# (e) Final answer
print("\n=== (e) FINAL ANSWER ===")
print(result["messages"][-1].content)


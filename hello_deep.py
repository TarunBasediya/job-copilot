from deepagents import create_deep_agent
from llm import get_model

model = get_model()

agent = create_deep_agent(
    model=model,
    # tools=[],
    system_prompt="You are a helpful assistant.",
)

result = agent.invoke(
    {
        "messages":[
            {
                "role":"user",
                "content":(
                     "Make a 3-step plan to learn React hooks. "
                    "Save the plan in a file called plan.md, "
                    "then tell me what you saved."
                    ,
                )
            }
        ]
    }
)

for m in result["messages"]:
    for tc in getattr(m,"tool_calls",None) or []:
        print("TOOL CALL:",tc["name"])

print("\n FINAL ANSWER", result["messages"][-1].content)
print("\n FILES", list((result.get("files") or {}).keys()))
print("TODOS", result.get("todos"))
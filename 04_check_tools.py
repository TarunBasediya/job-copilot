from deepagents import create_deep_agent
from llm import get_model

model = get_model()
agent = create_deep_agent(model=model, system_prompt="You are a helpful assistant.")

print("graph nodes:", list(agent.nodes.keys()))

try:
    names = sorted(agent.nodes["tools"].bound.tools_by_name.keys())
    print("tools available to the agent:")
    for n in names:
        print("  -", n)
except Exception as e:
    print("could not read the tool list this way:", repr(e))
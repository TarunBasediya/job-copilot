def file_text(value):
    # Version ke hisaab se file ka value string ya dict ho sakta hai
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, list):
            return "\n".join(content)
        if isinstance(content, str):
            return content
    return str(value)


def show_tool_calls(result):
    print("=== TOOL CALLS IN ORDER ===")
    for m in result["messages"]:
        for tc in getattr(m, "tool_calls", None) or []:
            print("-", tc["name"])


def show_todo_history(result):
    print("\n=== write_todos CALLS ===")
    call_no = 0
    for m in result["messages"]:
        for tc in getattr(m, "tool_calls", None) or []:
            if tc["name"] != "write_todos":
                continue
            call_no += 1
            print(f"\ncall #{call_no}:")
            todos = tc["args"].get("todos")
            if todos is None:
                # Field ka naam alag hua toh raw args dikha do
                print("  raw args:", tc["args"])
            else:
                for t in todos:
                    print(f"   [{t.get('status')}] {t.get('content')}")
    if call_no == 0:
        print("  (never called)")


def show_state(result):
    print("\n=== STATE ===")
    print("state keys:", list(result.keys()))
    print("todos:", result.get("todos"))
    for path, value in (result.get("files") or {}).items():
        print(f"\nFILE {path}")
        print(file_text(value)[:300])
from langchain_ollama import ChatOllama
from langsmith.schemas import Run, Example

judge_llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0
)

async def correct(outputs: dict, reference_outputs: dict) -> bool:
    instructions = (
        "Given an actual answer and an expected answer, determine whether "
        "the actual answer contains all of the information in the "
        "expected answer. Respond with 'CORRECT' or 'INCORRECT'."
    )

    actual_answer = outputs["messages"][-1].content
    expected_answer = reference_outputs["answer"]

    response = await judge_llm.ainvoke([
        {"role": "system", "content": instructions},
        {
            "role": "user",
            "content": f"ACTUAL ANSWER: {actual_answer}\n\nEXPECTED ANSWER: {expected_answer}"
        }
    ])

    return response.content.strip().upper() == "CORRECT"

def right_tool(outputs: dict) -> bool:
    tool_calls = outputs["messages"][1].tool_calls
    return bool(tool_calls and tool_calls[0]["name"] == "search")

def right_tool_from_run(run: Run, example: Example) -> dict:
    agent_run = next(r for r in run.child_runs if r.name == "agent")
    tool_calls = agent_run.outputs["messages"][-1].tool_calls
    return {
        "key": "right_tool",
        "value": bool(tool_calls and tool_calls[0]["name"] == "search")
    }
def agent_calls_search(outputs: dict) -> dict:
    """
    For node-level evaluation of the agent node ONLY.
    The agent node returns: {"messages": [AIMessage]}
    So we check tool_calls on that single AIMessage.
    """
    msg = outputs["messages"][-1]
    tool_calls = getattr(msg, "tool_calls", None) or []
    ok = bool(tool_calls and tool_calls[0].get("name") == "search")
    return {"key": "agent_calls_search", "value": ok}


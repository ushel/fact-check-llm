from typing import Literal
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, START, StateGraph
from langchain_ollama import ChatOllama

from agent.state import State
from agent.tools import TOOLS

tool_node = ToolNode(TOOLS)

model = ChatOllama(
    model="llama3.1:8b",
    temperature=0
).bind_tools(TOOLS)

def should_continue(state: State) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

def call_model(state: State):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def build_graph():
    workflow = StateGraph(State)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")

    return workflow.compile()

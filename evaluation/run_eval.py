# evaluation/run_eval.py
import asyncio
from langsmith import aevaluate
from langchain_core.runnables import RunnableLambda

from agent.graph import build_graph, call_model  # import call_model
from evaluation.dataset import create_dataset
from evaluation.evaluators import correct, right_tool, agent_calls_search  # new evaluator

def example_to_state(inputs: dict) -> dict:
    return {"messages": [{"role": "user", "content": inputs["question"]}]}

async def main():
    create_dataset()

    app = build_graph()
    target = example_to_state | app

    print("▶ Running full graph evaluation")
    await aevaluate(
        target,
        data="weather agent",
        evaluators=[correct, right_tool],
        max_concurrency=4,
        experiment_prefix="claude-3.5-baseline",
    )

    print("▶ Running node-level evaluation")

    # Node-level target: run ONLY the agent node function you wrote
    node_target = example_to_state | RunnableLambda(call_model)

    await aevaluate(
        node_target,
        data="weather agent",
        evaluators=[agent_calls_search],   # evaluator suited to agent-node output
        max_concurrency=4,
        experiment_prefix="claude-3.5-model-node",
    )

if __name__ == "__main__":
    asyncio.run(main())

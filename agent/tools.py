from langchain.tools import tool

@tool
def search(query: str) -> str:
    """Call to surf the web."""
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."

TOOLS = [search]

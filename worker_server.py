# convert 'predict_tool' to Claude (from OpenAI)
from mcp.server.fastmcp import FastMCP
from anthropic import Anthropic
import ast
import joblib
import numpy as np
import httpx
import asyncio
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from dotenv import load_dotenv
load_dotenv()

anthropic = Anthropic()
mcp = FastMCP("worker")

def predict_species(features: list[float]) -> str:
    print('Here in predict_species')
    """
    Predict the species of an iris flower given its measurements.
    Input format: [sepal_length, sepal_width, petal_length, petal_width]
    """
    target_names = ['setosa', 'versicolor', 'virginica']
    model_path = "iris_model.pkl"

    if not isinstance(features, list) or len(features) != 4:
        raise ValueError("Expected list of 4 numerical values.")

    try:
        with open(model_path, 'rb') as file:
            model = joblib.load(file)
    except Exception as ex:
        raise ValueError(f"Error loading model: {ex}")

    sample = np.array([features])
    prediction = model.predict(sample)
    print(f'prediction: {prediction}')
    return target_names[prediction[0]]

@mcp.tool()
async def predict_tool(query: str) -> str:
    """
    Predict the species of an iris flower using a natural language description of its measurements.

    Use this tool when the user provides a prompt like:
    "Predict the species of an iris flower with sepal length 5.1, sepal width 3.5, petal length 1.4, and petal width 0.2."

    The tool will extract the numerical measurements using a language model and call a local classifier.

    Args:
        query: A natural language string describing the iris measurements.

    Returns:
        The predicted species name (e.g., "setosa").
    """
    prompt = (
        "Extract the four numerical values from this sentence and return them "
        "as a Python list in this order: [sepal_length, sepal_width, petal_length, petal_width]. "
        "Only return the list, nothing else.\n\n"
        f"Sentence: {query}"
    )

    try:
        response = anthropic.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=100,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        list_str = response.content[0].text.strip()
        print("Claude response:", list_str)

        features = ast.literal_eval(list_str)
        if not isinstance(features, list) or len(features) != 4:
            return f"Error: Expected a list of 4 numbers, got: {features}"

        return predict_species(features)

    except Exception as e:
        return f"Error in predict_tool: {e}"

@mcp.tool()
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for a user-provided topic or phrase and return a snippet from the top result.

    Args:
        query: A natural language question or search term (e.g., "Marie Curie")

    Returns:
        The snippet from the first matching result on Wikipedia.
    """
    try:
        result = httpx.get("https://en.wikipedia.org/w/api.php", params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json"
        }).json()
        return result["query"]["search"][0]["snippet"]
    except Exception as e:
        return f"Error accessing Wikipedia: {e}"

@mcp.tool()
def tavily_search(query: str) -> list[str]:
    """
    Perform a web search for recent or current information using the Tavily API.

    Use this tool when the user asks for current events, recent news, or up-to-date developments
    that would not be found in Wikipedia or general knowledge.

    Args:
        query: A natural language search query. Example: "Latest breakthroughs in AI as of 2025"

    Returns:
        A list of result snippets (summaries or links) retrieved from the web using Tavily.
    """
    try:
        tavily_tool = TavilySearchResults(max_results=2)
        results = tavily_tool.run(query)
        return results
    except Exception as e:
        return [f"Error using Tavily: {e}"]

if __name__ == "__main__":
    # print("(3) worker_server started and listening...")

    # Quick standalone test
    print(predict_tool.__doc__)
    asyncio.run(predict_tool(
        "Predict the species of an iris flower with sepal length 6.3, sepal width 3.3, petal length 6, and petal width 2"))

    mcp.run(transport='stdio')

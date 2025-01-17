from langchain_community.tools.tavily_search import TavilySearchResults

from dotenv import load_dotenv
load_dotenv()

def search_on_tavily(query: str):
    """Search tavily for updated information."""
    search = TavilySearchResults()
    res = search.run(f"{query}")
    return res
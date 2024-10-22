import os
from langchain_community.tools.tavily_search import TavilySearchResults

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")


class WebSearchTool:
    web_search_tool = TavilySearchResults(k=3)

    @classmethod
    def search(self, query: str):
        return self.web_search_tool.invoke(query)

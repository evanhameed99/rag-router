from langchain_community.llms import Ollama
from models.ollamaLLMProvider import OllamaLLMProvider


class ModelLoader:
    def __init__(self, model_name, format=""):
        self.model_name = model_name
        self.model = OllamaLLMProvider(model_name, format)

    def invoke(self, query: str):
        return self.model.llm.invoke(query)

    async def astream(self, query: str):
        async for chunk in self.model.llm.astream(query):
            yield chunk

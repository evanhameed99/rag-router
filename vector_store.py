from langchain_chroma import Chroma
from embedding_model import EmbeddingModel


class VectorStore:
    def __init__(self, collection_name, embedding_function, persist_directory=None):
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_directory,  # Optional: only use if provided
        )

    def add_documents(self, docs):
        return self.vector_store.add_documents(documents=docs)

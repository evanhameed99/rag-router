import json
import asyncio
from data_loader import retrieve_data, extract_content_and_tags
from markdown_splitter import MarkdownSplitter
from embedding_model import EmbeddingModel
from vector_store import VectorStore
from models.model_loader import ModelLoader
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from prompt_templates import PromptTemplateFactory
from web_search_tool import WebSearchTool


MODEL_NAME = "llama3.2:3b-instruct-fp16"


# Example usage


async def main():

    # 1. Get the data
    blog_posts = retrieve_data()
    content, tags_list = extract_content_and_tags(blog_posts)

    # 2. Produce Chunks
    markdown_splitter = MarkdownSplitter(strip_headers=False)
    splitted_chunks = markdown_splitter.split_multiple_docs(content)

    # 3. Intitialize Embeddings Model
    embedding_model = EmbeddingModel()

    # 4. Intialize Vector Store

    vector_store = VectorStore("blog_posts", embedding_model, "./chroma_local_db")

    # 5. Initialize models
    model_json = ModelLoader(MODEL_NAME, format="json")
    model = ModelLoader(MODEL_NAME)

    # 6. Set up datasource router chain
    prompt_factory = PromptTemplateFactory(tag_list=tags_list)

    datasource_router_prompt_template = (
        prompt_factory.get_datasource_router_prompt_template()
    )
    data_source_router_chain = datasource_router_prompt_template | model_json.model.llm

    while True:
        # Get user query ...
        user_query = input("\nType your input here: ")
        # Exit conversation on submitting '/bye'
        if user_query == "/bye":
            print("See you!")
            break

        datasource_router_result = json.loads(
            data_source_router_chain.invoke(input={"input": user_query})
        )

        datasource = datasource_router_result["datasource"]
        print(datasource)
        context = None
        if datasource == "vectorstore":
            print("Vector store search ...")
            search_result = vector_store.vector_store.similarity_search(user_query)
            if search_result:
                context = search_result[0].page_content
        else:
            print("Web search ...")
            web_search_result = WebSearchTool.search(user_query)
            if web_search_result:
                context = web_search_result[0]["content"]

        print("Context: ", context)
        model_answer_prompt_template = prompt_factory.get_model_answer_prompt_template(
            user_query=user_query, context=context
        )

        model_answer_chain = model_answer_prompt_template | model.model.llm

        async for chunk in model_answer_chain.astream(
            input={"user_query": user_query, "context": context}
        ):
            print(chunk, end="", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nForce exit detected. Shutting down ...")

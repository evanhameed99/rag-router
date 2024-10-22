from langchain.prompts import ChatPromptTemplate


class PromptTemplateFactory:

    def __init__(self, tag_list=None):
        self.tag_list = tag_list

    def get_datasource_router_sys_message(self):
        topics = ", ".join(self.tag_list) if self.tag_list else ""
        return f"""You are an expert at routing a user question to a vectorstore or web search or zero.
        The vectorstore contains documents related to {topics}.                                    
        Use the vectorstore for questions on these topics. For all else, and especially for current events, use web-search.
        For simple questions that can be answered very easily such as greetings or thanking, simply use zero in the datasource.
        Return JSON with a single key, 'datasource', that is 'websearch' or 'vectorstore' or 'zero' depending on the question.
        """

    def get_datasource_router_prompt_template(self):
        datasource_router_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.get_datasource_router_sys_message()),
                ("human", "{input}"),
            ]
        )
        return datasource_router_prompt_template

    def get_model_answer_prompt_template(self, user_query: str, context: str):
        model_answer_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """ You are a helpful AI Assistant. Your goal is to help people with their inquiries.
                    Answer the following question with the help of the following context:\n{context}
                    """,
                ),
                ("human", "{user_query}"),
            ]
        )
        return model_answer_prompt_template

    def get_grader_prompt_template(self, user_query: str, context: str):
        grader_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a grader assessing the relevance of a retrieved document to a user question.
                    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
                    """,
                ),
                (
                    "human",
                    """Here is the retrieved document: \n\n {context} \n\n Here is the user question: \n\n {user_query}. 
                    Carefully and objectively assess whether the document contains at least some information that is relevant to the question.
                    Return JSON with a single key, 'binary_score', that is 'yes' or 'no' to indicate whether the document contains at least some information that is relevant to the question.""",
                ),
            ]
        )
        return grader_prompt_template

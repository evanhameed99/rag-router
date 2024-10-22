from langchain_text_splitters import MarkdownHeaderTextSplitter


class MarkdownSplitter:
    def __init__(
        self,
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ],
        strip_headers=True,
    ):
        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=strip_headers
        )

    def split_one_doc(self, doc):
        return self.splitter.split_text(doc)

    def split_multiple_docs(self, docs):
        chunks = []
        for doc in docs:
            chunks.extend(self.split_one_doc(doc))
        return chunks

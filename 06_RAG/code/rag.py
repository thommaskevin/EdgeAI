from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata


class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        # Initialize the Ollama model
        self.model = OllamaLLM(model="llama3.2:1b", base_url="http://127.0.0.1:11434")
        # Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        # Define the prompt template
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks. Use the following context to answer the question. 
            If you don't know the answer, simply say you don't know. Use at most three sentences and be concise in your response. [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )

    def ingest(self, pdf_file_path: str):
        try:
            # Load the PDF file
            docs = PyPDFLoader(file_path=pdf_file_path).load()
            # Split the documents into chunks
            chunks = self.text_splitter.split_documents(docs)
            # Filter out complex metadata
            chunks = filter_complex_metadata(chunks)

            # Create a vector store from the chunks
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=FastEmbedEmbeddings(),
                persist_directory="./chroma_db"  # Optional: Persist the vector store to disk
            )

            # Initialize the retriever
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 3,
                    "score_threshold": 0.5,
                },
            )

            # Set up the chain for querying
            self.chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.model
                | StrOutputParser()
            )
        except Exception as e:
            print(f"Error during ingestion: {e}")

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."

        try:
            return self.chain.invoke(query)
        except Exception as e:
            return f"Error during query processing: {e}"

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None

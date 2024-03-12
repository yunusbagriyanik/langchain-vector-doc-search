import os

from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain import OpenAI
import pinecone

load_dotenv()

if __name__ == "__main__":
    print("Langchain Vector Document Search")
    loader = TextLoader("FILE_PATH_REFERENCE")

    pinecone_instance = pinecone.Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    print(pinecone_instance.list_indexes())

    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(len(texts))

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    index = Pinecone.from_documents(
        texts, embeddings, index_name="pinecone_index_name"
    )

    # return_source_documents= True
    qa_retrieval = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=index.as_retriever()
    )

    query = "Query"
    result = qa_retrieval({"query": query})
    print(result)

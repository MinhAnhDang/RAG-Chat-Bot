import os
import getpass
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import pinecone
from pinecone import Pinecone, ServerlessSpec
# from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv
import time
from uuid import uuid4

load_dotenv()


def load_documents(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=300):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(documents=documents)
    return documents


def create_vector_database(INDEX_NAME, data_directory, embeddings):
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    
    if not os.getenv("PINECONE_API_KEY"):
        os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")
    pc = Pinecone(api_key=pinecone_api_key)
    
    # if does not exist, create index
    # spec = ServerlessSpec(cloud='aws', region="us-east-1-aws")
    pc.create_index(
        name=INDEX_NAME,
        dimension=768, # Replace with your model dimensions
        metric="cosine", # Replace with your model metric
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ) 
    )
    
    # wait for index to be initialized
    while not pc.describe_index(INDEX_NAME).status['ready']:
        time.sleep(1)

    # connect to index
    index = pc.Index(INDEX_NAME)
    time.sleep(1)
    # view index stats
    index.describe_index_stats()
    

    documents = load_documents(data_directory)
    documents = split_documents(documents)
    print("Documents to be embedded:", documents)

    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents)
    
    return vector_store



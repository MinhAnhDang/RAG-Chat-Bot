from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

import streamlit as st
from streamlit_chat import message
from pinecone import Pinecone, ServerlessSpec
import os
import time
from indexing import create_vector_database
from utils import *
from dotenv import load_dotenv
load_dotenv()

INDEX_NAME = "langchain-chatbot"
data_directory = "data/"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


#Initialize vector store
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
exist_indexes = [index['name'] for index in pc.list_indexes()]
if pc.has_index(INDEX_NAME):
    index = pc.Index(INDEX_NAME)
    time.sleep(1)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
else:
    print("Do not have database. Creating from our data...")
    vector_store = create_vector_database(INDEX_NAME, data_directory, embeddings)
    
# Turn vector_store into retriever
retriever = vector_store.as_retriever(search_type="similarity",
                                      search_kwargs={'k':3})  
         
st.subheader("Chatbot with Langchain, ChatGPT, Pinecone, and Streamlit")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]
    
if 'requests' not in st.session_state:
    st.session_state['requests'] = []
    
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


st.title("Langchain Chatbot")
# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()


with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        print(query)
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            # st.code(conversation_string)
            # print(context)  
            response = get_response(conversation=conversation_string, query=query, retriever=retriever)
        st.session_state.requests.append(query)
        st.session_state.responses.append(response) 
with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')



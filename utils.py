from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_pinecone import PineconeVectorStore
import pinecone
from pinecone import Pinecone
import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')
llm = HuggingFaceEndpoint(  
                        repo_id="microsoft/Phi-3-mini-4k-instruct",  
                        task="text-generation",  
                        max_new_tokens=512,  
                        do_sample=False,  
                        repetition_penalty=1.03,  
                        )  
model = ChatHuggingFace(llm=llm, verbose=True)

def get_response(conversation, query, retriever):
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
        )
       
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
        
    ])
    history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)
    
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. Use "
        "the following pieces of retrieved context to answer the "
        "question. If you don't know the answer, just say that you "
        "don't know. Use three sentences maximum and keep the answer "
        "concise."
        "\n\n"
        "{context}"
        )

   
    # Create a prompt template for answering questions
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

    # Create a retrieval chain that combines the history-aware retriever and the question answering chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
    result = rag_chain.invoke({"input": query, "chat_history": conversation})
    return result

def get_conversation_string():
    conversation_string = ""
    
    for i in range(len(st.session_state['responses'])-1):
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string

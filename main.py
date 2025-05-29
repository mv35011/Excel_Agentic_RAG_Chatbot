import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from llm_mechs import RAGSystem
import time
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Streamlit Bot", page_icon="🤖")
st.title("Streaming Bot")

rag = RAGSystem(groq_api_key)

if rag.process_doc("S&M Data.xlsx"):
    print("Chatbot running...")
    print("Data preview:")
    if rag.df is not None:
        print(f"Shape: {rag.df.shape}")
        print(f"Columns: {list(rag.df.columns)}")
        if 'Region' in rag.df.columns:
            print("Regional distribution:")
            print(rag.df['Region'].value_counts())
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
    user_query = st.chat_input("Your message")
    if user_query is not None and user_query.strip() != "":
        st.session_state.chat_history.append(HumanMessage(user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        chat_history_text = [msg.content for msg in st.session_state.chat_history if isinstance(msg, AIMessage)]

        with st.chat_message("AI"):
            ai_response = str(rag.query(user_query)['response']) + "\n" + "," + str(
                rag.query(user_query)['sources']) + "\n" + "," + str(rag.query(user_query)['confidence'])
            st.markdown(ai_response)

        st.session_state.chat_history.append(AIMessage(ai_response))


else:
    print("Failed to process the document. Please check the file path and format.")

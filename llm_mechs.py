import pandas as pd
import numpy as np
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
import os
from langchain.memory import ConversationBufferWindowMemory
from pydantic import BaseModel
from langchain.schema import Document
from typing import List, Dict, Any, Optional
import json
import chromadb
from langchain.schema.output_parser import StrOutputParser
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from datetime import datetime
from langchain.chains import RetrievalQA
from langchain.embeddings import OllamaEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")

class RAGSystem:
    # Initailizing the llm, vectordb, agent executer, memory buffer(for streamlit app if made), qa_chain for fallback
    def __init__(self, groq_api_key):
        # self.embeddings = ollamaEmbeddings("llama2)
        # self.embeddings = HuggingFaceBgeEmbeddings(
        #     model_name="sentence-transformers/all-MiniLM-L6-v2",
        #     model_kwargs={'device': 'cpu'}
        # )
        self.embeddings = CohereEmbeddings(model="embed-english-light-v3.0", cohere_api_key=cohere_api_key)
        self.llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            groq_api_key=groq_api_key,
            temperature=0
        )

        # self.llm = ChatOllama(model="llama2")
        self.vectorstores = None
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection_name = "business_data"
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.retriever = None
        self.agent_executor = None
        self.qa_chain = None
        self.df = None

        # prompt template for the initial analysis of the data queried from the db
        # some errors came in previous prototypes so made the model to do some extra countings and rechecks

        self.analysis_prompt = PromptTemplate.from_template("""
          You are a business intelligence analyst. Analyze the provided data carefully and answer with specific numbers and facts.

          Context from business data:
          {context}

          User Query: {question}
          Chat History: {chat_history}

          CRITICAL INSTRUCTIONS:
          1. Count and analyze the data PRECISELY - double-check your calculations
          2. When counting visits, customers, or any metrics, be extremely careful with your arithmetic
          3. For regional analysis, make sure you're grouping by the correct region field
          4. For person-based queries, group by the correct name/person field
          5. Always provide specific numbers and cite the exact data points
          6. If asked about "most" or "least", compare ALL options and provide rankings
          7. Show your work - explain how you arrived at the numbers
          8. If the context doesn't contain enough data, request more specific information

          Format your response as:
          **Answer:** [Direct answer with specific number]
          **Calculation:** [Show how you calculated this]
          **Supporting Data:** [List the relevant data points you used]

          Response:
      """)

        self.summarization_prompt = PromptTemplate.from_template("""
          Based on the following business visit data, provide a comprehensive summary:

          Data: {context}

          Focus on:
          - Key customer interactions and outcomes
          - Sales performance and order bookings
          - Outstanding issues and follow-ups needed
          - Regional performance patterns
          - Product division insights

          Provide a well-structured, professional summary with specific numbers and metrics.
      """)

    # using pandas to open the excel, cleaned data(extra spaces for missing values converting dates to date time values), splits the document with recursive
    # splitter(chunk size and overlap to be adjusted), then converting to Documents and storing to vector db with embedding functions

    def process_doc(self, file_path: str) -> bool:
        try:
            self.df = pd.read_excel(file_path)
            print("Excel loaded successfully...")
            print(f"Data shape: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")

            self.df = self.clean_df(self.df)

            documents = self.create_documents_from_df(self.df)

            doc_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=150,
                separators=["\n\n", "\n", ".", " "]
            )
            split_text = doc_splitter.split_documents(documents)

            try:
                self.chroma_client.delete_collection(self.collection_name)
            except:
                pass

            self.vectorstores = Chroma.from_documents(
                documents=split_text,
                embedding=self.embeddings,
                client=self.chroma_client,
                collection_name=self.collection_name,
                persist_directory="./chroma_db"
            )

            self.retriever = self.vectorstores.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10}  # Retrieve more documents for better context
            )

            self.setup_agent_tools()
            return True

        except Exception as e:
            print(f"Error processing document: {e}")
            return False

    def clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.fillna("")

        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass

        return df

    def create_documents_from_df(self, df: pd.DataFrame) -> List[Document]:
        documents = []

        self.create_summary_documents(df, documents)

        for idx, row in df.iterrows():
            content_parts = []
            metadata = {"row_id": idx, "type": "individual_record"}

            for col, value in row.items():
                if pd.isna(value) or value == "":
                    continue

                content_parts.append(f"{col}: {value}")

                if col.lower() in ['customer', 'region', 'product division', 'owner name', 'visit date']:
                    metadata[col.lower().replace(" ", "_")] = str(value)

            content = "\n".join(content_parts)

            documents.append(Document(
                page_content=content,
                metadata=metadata
            ))

        return documents

    # initial project was halucinating with the regional and person data, so this function creates document summary along with the metadatas for model to directly extract regional data
    # basically checks for region related data in df column and converts into human redable string summary

    def create_summary_documents(self, df: pd.DataFrame, documents: List[Document]):

        if 'Region' in df.columns:
            region_counts = df['Region'].value_counts()
            region_summary = "Regional Visit Distribution:\n"
            for region, count in region_counts.items():
                region_summary += f"- {region}: {count} visits\n"

            documents.append(Document(
                page_content=region_summary,
                metadata={"type": "regional_summary"}
            ))

        owner_columns = [col for col in df.columns if 'owner' in col.lower() or 'name' in col.lower()]
        for col in owner_columns:
            if col in df.columns:
                person_counts = df[col].value_counts()
                person_summary = f"{col} Visit Distribution:\n"
                for person, count in person_counts.items():
                    if person and str(person).strip():
                        person_summary += f"- {person}: {count} visits\n"

                documents.append(Document(
                    page_content=person_summary,
                    metadata={"type": f"{col.lower()}_summary"}
                ))

    # setting up the agentic tools and agent
    def setup_agent_tools(self):
        def direct_data_analysis(query: str) -> str:

            try:

                query_lower = query.lower()

                if not isinstance(self.df, pd.DataFrame):
                    return "Data not properly loaded"

                if "region" in query_lower and ("visit" in query_lower or "count" in query_lower):
                    if 'Region' in self.df.columns:
                        region_counts = self.df['Region'].value_counts()
                        result = "Regional visit counts:\n"
                        for region, count in region_counts.items():
                            result += f"- {region}: {count} visits\n"

                        if "west" in query_lower:
                            west_count = region_counts.get('West', 0)
                            result += f"\nSpecific answer: West region has {west_count} visits"
                        elif "least" in query_lower:
                            min_region = region_counts.idxmin()
                            min_count = region_counts.min()
                            result += f"\nRegion with least visits: {min_region} ({min_count} visits)"

                        return result

                if ("who" in query_lower or "person" in query_lower) and "most" in query_lower:
                    owner_columns = [col for col in self.df.columns if 'owner' in col.lower() or 'name' in col.lower()]
                    results = []

                    for col in owner_columns:
                        if col in self.df.columns:
                            person_counts = self.df[col].value_counts()
                            if not person_counts.empty:
                                top_person = person_counts.index[0]
                                top_count = person_counts.iloc[0]
                                results.append(f"{col}: {top_person} with {top_count} visits")

                    return "Most visits by person:\n" + "\n".join(results)

                docs = self.retriever.get_relevant_documents(query)
                context = "\n\n".join([doc.page_content for doc in docs[:8]])
                return f"Retrieved context:\n{context}"

            except Exception as e:
                return f"Error in direct analysis: {str(e)}"

        # retreiving the data from the vector store
        def enhanced_search(query: str) -> str:
            try:

                docs = self.retriever.get_relevant_documents(query)
                if not docs:
                    return "No relevant documents found"

                results = []
                for doc in docs[:8]:
                    results.append(doc.page_content)

                return "\n\n".join(results)
            except Exception as e:
                return f"Failed to search documents: {str(e)}"

        # using the analysis prompt template to get the llm to analyze the retrieved data
        def analyze_with_llm(query: str) -> str:
            try:
                direct_result = direct_data_analysis(query)

                docs = self.retriever.get_relevant_documents(query)
                context = "\n\n".join([doc.page_content for doc in docs[:8]])

                full_context = f"Direct Data Analysis:\n{direct_result}\n\nAdditional Context:\n{context}"

                analysis_chain = self.analysis_prompt | self.llm | StrOutputParser()
                response = analysis_chain.invoke({
                    "context": full_context,
                    "question": query,
                    "chat_history": ""
                })
                return response
            except Exception as e:
                return f"Failed to analyze: {str(e)}"

        # tools array for the agent
        tools = [
            Tool(
                name="Direct Data Analysis",
                func=direct_data_analysis,
                description="Perform direct counting and analysis on the raw data for accurate metrics"
            ),

            Tool(
                name="Enhanced Document Search",
                func=enhanced_search,
                description="Search through processed documents with enhanced context"
            ),

            Tool(
                name="LLM Analysis",
                func=analyze_with_llm,
                description="Comprehensive analysis combining direct data analysis with LLM reasoning"
            )
        ]
        # template for the agent to use and getting it to use direct analysis for preventing hallucinations
        self.agent_prompt = PromptTemplate.from_template("""
            You are a business intelligence agent with access to comprehensive business data.

            IMPORTANT: For counting and statistical queries, ALWAYS use "Direct Data Analysis" first to get accurate numbers.

            You have access to the following tools:
            {tools}

            Use the following format:
            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Question: {input}
            {agent_scratchpad}
        """)
        # creating the agent and specifying the buffer memory, max iteration to be twicked for appropreate output- less for less api overhead and more for bypassign max iteration limit llm might reach

        try:
            agent = create_react_agent(self.llm, tools, self.agent_prompt)
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                max_iterations=8,
                memory=self.memory,
                handle_parsing_errors=True
            )

        except Exception as e:
            print(f"Failed to create the agent: {str(e)}")
            self.fallback_chain()

    # fallback chainc incase agent fails to be created
    def fallback_chain(self):

        try:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.analysis_prompt}
            )
        except Exception as e:
            print(f"Failed to create fallback chain: {str(e)}")

    # query function for initializing execution and output format of the rag
    def query(self, question: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        try:
            if not self.vectorstores:
                return {
                    "response": "Please upload and process an Excel File",
                    "sources": [],
                    "confidence": 0.0
                }

            history_text = ""
            if chat_history:
                history_text = "\n".join([
                    f"Human: {item.get('human', '')}\nAI: {item.get('ai', '')}"
                    for item in chat_history[-3:]
                ])

            if self.agent_executor:
                result = self.agent_executor.invoke({
                    "input": question,
                    "chat_history": history_text
                })
                response = result.get("output", "Couldn't process the query")
                sources = ["Agent-based analysis"]
                confidence = 0.9
            else:
                if not self.qa_chain:
                    self.fallback_chain()

                if self.qa_chain:
                    result = self.qa_chain({"query": question})
                    response = result["result"]
                    sources = [doc.metadata.get("source", "Business Data")
                               for doc in result.get("source_documents", [])]
                    confidence = 0.7
                else:
                    response = "System not properly initialized"
                    sources = []
                    confidence = 0.0

            return {
                "response": response,
                "sources": sources,
                "confidence": confidence
            }

        except Exception as e:
            return {
                "response": f"An error occurred: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }


if __name__ == "__main__":
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

        while True:
            user_input = input("\nUser: ")
            if user_input.lower() in ["close", "end", "end chat"]:
                break

            result = rag.query(user_input)
            print(f"\nResponse: {result['response']}")
            print(f"Sources: {', '.join(result['sources'])}")
            print(f"Confidence: {result['confidence']}")
            print("-" * 50)
    else:
        print("Failed to process the document. Please check the file path and format.")

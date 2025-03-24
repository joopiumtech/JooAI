import os
import re
import ast
import openai
import pygame
import time

from agents.tavily_search_agent import tavily_search
from utils import fetch_restaurant_name, initialize_db, store_merchant_memory, get_merchant_memory
from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Load environment variables
from dotenv import load_dotenv
load_dotenv(override=True)


# Initialize the LLM model
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_retries=1,
    api_key=os.environ.get("OPENAI_API_KEY"),
    streaming=True
)


def text_to_speech(text, filename="audio_response/response.mp3"):
    """
    Converts text to speech using OpenAI TTS and plays it.
    """
    response = openai.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "wb") as f:
        f.write(response.content)

    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)  # Small delay to allow playback to finish

    # Stop and unload music to allow overwriting
    pygame.mixer.music.stop()
    pygame.mixer.quit()


def get_business_reference_data(query: str):
    # Initialize pinecone
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = os.environ.get("INDEX_NAME")
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=os.environ.get("PINECONE_ENVIRONMENT")),
        )

    index = pc.Index(index_name)

    # Initialize embedding model
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=os.environ.get('OPENAI_API_KEY')
    )

    vector_store = PineconeVectorStore(index=index, embedding=embedding_model)

    results = vector_store.similarity_search(
        query,
        k=5,
        namespace="marketing_plan"
    )
    for res in results:
        return res.page_content
    


def query_db_for_merchant(query: str = None, audio_query: bool = False):
    """
    Authenticates a merchant and processes a text or audio query using an LLM-powered agent.

    Args:
        query (str, optional): The text query to be executed. Not required if using audio input.
        audio_query (bool, optional): If True, records an audio query, transcribes it, and then processes it.

    Returns:
        dict: A dictionary containing the AI-generated response.
    """
    try:        
        if not query:
            return {"ai_response": "No query provided."}

        query = re.sub(r"[\"'/]", " ", query)
        
        # Initialize database
        db = initialize_db()
        email = os.environ.get("ADMIN_EMAIL")

        """Creates and returns an agent executor configured for the database."""
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        tools = toolkit.get_tools()
        
        # Retrieve memory context
        chat_history = get_merchant_memory(email=email) or "[]"
        chat_history = ast.literal_eval(chat_history)

        # Process chat history
        memory_context = "\n".join(
            [f"merchant_query: {merchant_query}\nai_response: {ai_response}\n" for merchant_query, ai_response in chat_history]
        )

        RESTAURANT_NAME = fetch_restaurant_name()
        reference_data = get_business_reference_data(query=query)

        prompt_template = f"""Restaurnat Name: {RESTAURANT_NAME}
        You are an AI assistant for {RESTAURANT_NAME}, interacting with its SQL database to answer queries based on:
        Chat History: {memory_context}
        User Query: {query}

        Your goal is to provide accurate, concise, and insightful answers based on the restaurant's data.

        Query Execution Guidelines
        Database Understanding:
        First, retrieve and examine table structures before generating queries.
        Identify relevant tables and fields.
        For business-related queries, refer to {reference_data}.
        
        SQL Query Construction:
        Generate valid {{dialect}} SQL queries.
        Select only necessary columns, avoid SELECT *.
        Limit results to {{top_k}}, unless specified otherwise.
        Sort results meaningfully.
        Use British pounds (£) for monetary values, converting when necessary.

        Execution & Error Handling:
        Verify queries before execution and refine if errors occur.
        For bookings, return only latest records or say: "Currently, there are no booking records available."
        Answer directly when possible; otherwise, say "I don't know" —never guess.
        For queries related to "total sales," retrieve only records where status = 1 and calculate the total sum. 

        Restrictions:
        Read-Only Access – No INSERT, UPDATE, DELETE, or DROP.
        Use only approved database tools for responses.
        """
        
        system_message = prompt_template.format(dialect="mysql", top_k=5)
        agent_executor = create_react_agent(llm, tools, prompt=system_message)

        response = agent_executor.invoke({"messages": [{"role": "user", "content": query}]})
        final_answer = response["messages"][-1].content.strip()
        final_answer_db_values = re.sub(r"[\"'\\]", lambda m: {"'": "''", '"': '""', "\\": "\\\\"}[m.group()], final_answer)
        
        # If the AI cannot retrieve an answer, use external search
        if any(phrase in final_answer for phrase in ["I cannot retrieve", "I don't know", "I don't have"]):
            tavily_response = tavily_search(input=query)
            tavily_response_db_values = re.sub(r"[\"'\\]", lambda m: {"'": "''", '"': '""', "\\": "\\\\"}[m.group()], tavily_response)

            store_merchant_memory(email=email, merchant_query=query, ai_response=tavily_response_db_values)
            response_text = tavily_response
        else:
            store_merchant_memory(email=email, merchant_query=query, ai_response=final_answer_db_values)
            response_text = final_answer

        
        if audio_query:
            text_to_speech(text=response_text)
            return {"ai_response": response_text}
        else:
            return {"ai_response": response_text}

    except Exception as error:
        return {"ai_response": f"Oops! Something went wrong. Please try again.\n{error}"}
        
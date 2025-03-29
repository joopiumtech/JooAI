import os
import re
import openai
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI

# Import utilities
from agents.tavily_search_agent import tavily_search
from utils import fetch_restaurant_details, initialize_db, store_merchant_memory, get_merchant_memory, initialize_pinecone

# Load environment variables
load_dotenv(override=True)

# Pre-compile regex patterns
QUERY_CLEANER = re.compile(r"[\"'/]")
ESCAPE_DB_VALUES = re.compile(r"[\"'\\]")

# Global initializations
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0, max_retries=1, api_key=os.getenv("OPENAI_API_KEY"), streaming=True, max_completion_tokens=500)
db = initialize_db()
vector_store = initialize_pinecone()


def sanitize_query(query: str) -> str:
    """Sanitize user input for safer processing."""
    return QUERY_CLEANER.sub(" ", query)


def escape_db_values(text: str) -> str:
    """Escape special characters in text for database storage."""
    return ESCAPE_DB_VALUES.sub(lambda m: {"'": "''", '"': '""', "\\": "\\\\"}[m.group()], text)


def text_to_speech(text, filename="audio_response/response.mp3"):
    """Convert text to speech and play it."""
    import pygame
    import time
    
    response = openai.audio.speech.create(model="tts-1", voice="alloy", input=text)
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        f.write(response.content)

    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

    pygame.mixer.music.stop()
    pygame.mixer.quit()


def get_business_reference_data(query: str) -> str:
    """Retrieve relevant business data from Pinecone."""
    results = vector_store.similarity_search(query, k=5, namespace="marketing_plan")
    return results[0].page_content if results else ""



def get_drinks_reference_data(query: str) -> str:
    """Retrieve relevant drinks data from Pinecone."""
    results = vector_store.similarity_search(query, k=5, namespace="drinks_menu")
    return results[0].page_content if results else ""


def get_desert_reference_data(query: str) -> str:
    """Retrieve relevant desert data from Pinecone."""
    results = vector_store.similarity_search(query, k=5, namespace="desert_menu")
    return results[0].page_content if results else ""


def query_db_for_merchant(query: str = None, audio_query: bool = False):
    """Process merchant queries with database interactions and AI assistance."""
    if not query:
        return {"ai_response": "No query provided."}

    query = sanitize_query(query)
    email = os.getenv("ADMIN_EMAIL")

    # Fetch restaurant and memory context
    restaurant_details = fetch_restaurant_details()
    chat_history = get_merchant_memory(email) or "[]"

    # Process chat history (without ast.literal_eval for speed)
    memory_context = "\n".join(
        f"merchant_query: {entry[0]}\nai_response: {entry[1]}\n" for entry in eval(chat_history)
    )

    business_reference_data = get_business_reference_data(query)
    drinks_reference_data = get_drinks_reference_data(query)
    desert_reference_data = get_desert_reference_data(query)

    # Construct prompt
    system_message = f"""Restaurnat Name: {restaurant_details["restaurant_name"]}
    Restaurant Contact Number: {restaurant_details["contact_no"]}
    Restaurant Address: {restaurant_details["address"]}

    You are an AI assistant for {restaurant_details["restaurant_name"]}, interacting with its SQL database to answer restaurant-related queries.
    chat_history: {memory_context}

    Query Execution Guidelines:
    Understand the Database:
    First, retrieve and review table structures before constructing queries.
    Use only relevant tables and columns.

    SQL Query Construction:
    Generate syntactically correct {{dialect}} queries.
    Default to {{top_k}} results unless specified.
    Order results logically for clarity.
    Show monetary values in British pounds (£), converting if needed.

    Execution & Error Handling:
    Verify queries before execution.
    Refine and retry if errors occur.
    For bookings, provide only the latest data or state: "Currently, there are no booking records available."
    Refer to {business_reference_data}, {drinks_reference_data}, and {desert_reference_data} for relevant queries.
    Check the orders.other_info column for allergy details.
    If the answer is unknown, strictly respond with: "I don’t know."

    Restrictions:
    Read-only access—no INSERT, UPDATE, DELETE, or DROP queries.
    Use only provided tools and data.
    
    Your goal is to deliver accurate, concise, and insightful responses based on restaurant data.
    """

    # Initialize agent executor
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent_executor = create_react_agent(llm, toolkit.get_tools(), prompt=system_message)

    try:
        response = agent_executor.invoke({"messages": [{"role": "user", "content": query}]})
        final_answer = response["messages"][-1].content.strip()

        if "I cannot retrieve" in final_answer or "I don't know" in final_answer or "I don't have" in final_answer:
            response_text = tavily_search(input=query)
        elif "I encountered an issue" in final_answer:
            return {"ai_response": "Oops! Something went wrong. Please try again."}
        else:
            response_text = final_answer

        # Store response in memory
        store_merchant_memory(email=email, merchant_query=query, ai_response=escape_db_values(response_text))

        # Play audio if requested
        if audio_query:
            text_to_speech(response_text)
        
        return {"ai_response": response_text}

    except Exception:
        return {"ai_response": "Oops! Something went wrong. Please try again."}

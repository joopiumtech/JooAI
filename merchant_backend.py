import os
import re
import openai
import asyncio
import concurrent.futures
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
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0, max_retries=1, api_key=os.getenv("OPENAI_API_KEY"), streaming=True, max_completion_tokens=300)  # Reduced token limit
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


async def fetch_vector_data(query: str):
    """Fetch all vector data in parallel."""
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        tasks = [
            loop.run_in_executor(pool, lambda: vector_store.similarity_search(query, k=3, namespace="marketing_plan")),
            loop.run_in_executor(pool, lambda: vector_store.similarity_search(query, k=3, namespace="drinks_menu")),
            loop.run_in_executor(pool, lambda: vector_store.similarity_search(query, k=3, namespace="desert_menu")),
        ]
        results = await asyncio.gather(*tasks)
    return [r[0].page_content if r else "" for r in results]


async def query_db_for_merchant(query: str = None, audio_query: bool = False):
    """Process merchant queries with optimized performance."""
    if not query:
        return {"ai_response": "No query provided."}

    query = sanitize_query(query)
    email = os.getenv("ADMIN_EMAIL")

    # Fetch restaurant and memory context asynchronously
    restaurant_details_task = asyncio.to_thread(fetch_restaurant_details)
    chat_history_task = asyncio.to_thread(get_merchant_memory, email)
    vector_data_task = fetch_vector_data(query)

    restaurant_details, chat_history, vector_data = await asyncio.gather(
        restaurant_details_task, chat_history_task, vector_data_task
    )

    chat_history = chat_history or "[]"
    memory_context = "\n".join(f"merchant_query: {entry[0]}\nai_response: {entry[1]}\n" for entry in eval(chat_history))

    business_reference_data, drinks_reference_data, desert_reference_data = vector_data

    # Construct optimized system message
    system_message = f"""
    Restaurant: {restaurant_details["restaurant_name"]}
    Restaurant Contact: {restaurant_details["contact_no"]}
    Restaurant Address: {restaurant_details["address"]}
    chat_history: {memory_context}

    Guidelines for Query Execution:
    Understanding the Database:
    - Before generating any query, first retrieve and examine the available tables to understand what data can be accessed.
    - Identify the most relevant tables and check their schema before constructing your query.
    - Always represent monetary values in British pounds (£). If a value is given in another currency, convert it to pounds (£) using the most recent exchange rate. Clearly indicate the conversion when applicable. Never use dollars ($) or any other currency unless explicitly requested.

    Constructing SQL Queries:
    - Generate only syntactically correct {{dialect}} queries.
    - Focus only on relevant columns instead of selecting all columns from a table.
    - Unless the user specifies a particular number of results, limit queries to {{top_k}} results for efficiency.
    - When applicable, order results by a relevant column to provide the most insightful answers.
    
    Execution & Error Handling:
    - Always double-check your query before execution.
    - If an error occurs, refine the query and retry instead of returning incorrect results.
    - For queries related to bookings, retrieve only the most up-to-date information from the bookings table. If no data is available, respond strictly with: "Currently, there are no booking records available." 
    - For business-related queries, refer to {business_reference_data}.
    - For drinks-related queries, refer to {drinks_reference_data}.
    - For desert-related queries, refer to {desert_reference_data}.
    - If the query pertains to allergies, refer to the orders table and examine the other_info column for any recorded allergy information.
    - If you can answer it directly, do so.
    - If you don't know the answer, strictly respond with "I don't know". Don't try to create an answer from the data.

    Restrictions:
    - Do NOT execute any DML (INSERT, UPDATE, DELETE, DROP, etc.) operations—your role is strictly read-only.
    - Only use the tools provided to interact with the database and rely solely on the returned data to construct responses.
    
    Your goal is to provide accurate, concise, and insightful answers based on the restaurant's data.
    """

    # Initialize agent executor
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent_executor = create_react_agent(llm, toolkit.get_tools(), prompt=system_message)

    try:
        response = await asyncio.to_thread(agent_executor.invoke, {"messages": [{"role": "user", "content": query}]})
        final_answer = response["messages"][-1].content.strip()

        if "I don't know" in final_answer or "I don't have" in final_answer or "I cannot retrieve" in final_answer:
            response_text = await asyncio.to_thread(tavily_search, input=query)
        elif "I encountered an issue" in final_answer:
            return {"ai_response": "Oops! Something went wrong. Please try again."}
        else:
            response_text = final_answer

        # Store response in memory asynchronously
        await asyncio.to_thread(store_merchant_memory, email, query, escape_db_values(response_text))

        # Play audio asynchronously
        if audio_query:
            await asyncio.to_thread(text_to_speech, response_text)

        return {"ai_response": response_text}

    except Exception:
        return {"ai_response": "Oops! Something went wrong. Please try again."}

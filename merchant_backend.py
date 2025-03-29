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
    Contact: {restaurant_details["contact_no"]} | Address: {restaurant_details["address"]}
    chat_history: {memory_context}

    Query Execution:
    - Retrieve table structures before constructing queries.
    - Generate proper {{dialect}} queries.
    - Show monetary values in £ if needed.
    - Order results logically.
    - For bookings, provide only the latest data or state: "No bookings available."
    - Use {business_reference_data}, {drinks_reference_data}, {desert_reference_data} for context.
    - If unsure, reply: "I don’t know."

    Restrictions:
    - Read-only DB access. No INSERT, UPDATE, DELETE, DROP.
    """

    # Initialize agent executor
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent_executor = create_react_agent(llm, toolkit.get_tools(), prompt=system_message)

    try:
        response = await asyncio.to_thread(agent_executor.invoke, {"messages": [{"role": "user", "content": query}]})
        final_answer = response["messages"][-1].content.strip()

        if "I don’t know" in final_answer or "I cannot retrieve" in final_answer:
            response_text = await asyncio.to_thread(tavily_search, input=query)
        elif "I encountered an issue" in final_answer:
            return {"ai_response": "Oops! Something went wrong. Please try again."}
        else:
            response_text = final_answer

        # Store response in memory asynchronously
        await asyncio.to_thread(store_merchant_memory, email, query, escape_db_values(response_text))

        # # Play audio asynchronously
        # if audio_query:
        #     await asyncio.to_thread(text_to_speech, response_text)

        return {"ai_response": response_text}

    except Exception:
        return {"ai_response": "Oops! Something went wrong. Please try again."}

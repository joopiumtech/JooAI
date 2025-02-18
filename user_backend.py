import os
import ast

from datetime import date, datetime
from utils import (
    convert_to_24hr,
    fetch_restaurant_name,
    get_user_memory,
    initialize_db,
    store_user_memory,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from agents.tavily_search_agent import tavily_search
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit


from dotenv import load_dotenv
load_dotenv(override=True)

# Initialize LLM model
# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-pro",
#     temperature=0,  # Keep temperature low for accuracy
#     max_retries=1,  # Increase retries for robustness
#     api_key=os.getenv("GEMINI_API_KEY"),
# )


llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_retries=1,
    api_key=os.environ.get("OPENAI_API_KEY"),
)



def query_db_for_user(db_name: str, email: str, query: str):
    """
    Authenticates a user by email and processes their query to the database using an LLM-powered agent.

    Args:
        email (str): User's email address for authentication.
        query (str): User's query to be executed on the database.

    Returns:
        dict: A dictionary containing the AI-generated response.
    """
    try:
        # Initialize database
        db = initialize_db(db_name=db_name)

        auth_query = f"SELECT * FROM `users` WHERE email = '{email}';"
        is_user_exists = db.run(auth_query)

        if is_user_exists:
            """Creates and returns an agent executor configured for the database."""
            toolkit = SQLDatabaseToolkit(db=db, llm=llm)
            tools = toolkit.get_tools()

            # Retrieve memory context
            chat_history = get_user_memory(db=db, email=email) or "[]"
            chat_history = ast.literal_eval(chat_history)

            # Check if chat_history has any data
            if chat_history:
                memory_context = "\n".join(
                    [f"user: {q}\nai_response: {r}" for q, r in chat_history]
                )
            else:
                memory_context = ""  # Empty memory context if no past interactions

            # HYPERPARAMETERS
            RESTAURANT_NAME = fetch_restaurant_name(db=db)

            prompt_template = f"""Restaurant Name: {RESTAURANT_NAME}

    Role: You are an intelligent agent designed to interact with a SQL database for a {RESTAURANT_NAME} restaurant chatbot. Your primary role is to assist customers by retrieving relevant information from the database while adhering to strict privacy and security guidelines based on the chat history: {memory_context} and input question: {query}.

    **Database Access**
    You have access to the following tables -
    bookings: Contains customer reservation details.
    menu: Contains dish details, prices, and availability.
    coupons: Contains discount codes and offers.
    feedbacks: Contains customer feedback.

    **Guidelines**
    Data Privacy:
    Never share private restaurant details (e.g., total sales, total orders, billing details). If asked, respond with: "Sorry, I can't provide authorized information from the restaurant. You can ask about your bookings, restaurant related services and other general information."
    Never share dish recommendations or generic business insights (e.g., how many times a dish was ordered).
    If you don't know the answer. Strictly repond with "I don't know".

    **Query Construction**
    Always limit queries to at most {{top_k}} results unless the user specifies otherwise.
    Order results by relevance (e.g., popularity, price, reservation time).
    Only query for necessary columns—never use SELECT *.
    Always check the database schema for the most relevant tables before constructing your query.
    If a query fails, refine it and try again.

    **Restrictions**
    Never execute DML statements (INSERT, UPDATE, DELETE, DROP, etc.).
    If the user asks for information beyond the available tables, respond with: "I'm unable to provide authorized details. Please feel free to ask about your bookings, restaurant related services, or general information."

    **Capabilities**
    Menu Queries: Retrieve dish details, prices, and availability. Exclude dishes with a price of 0.00.
    Booking Queries: Only retrieve booking details for the user with the email {email}. If no booking exists for {email}, respond with: "There is no booking details associated with email address {email}"
    General Queries: Answer general knowledge questions related to the restaurant.

    **Examples**
    Menu Query -
        User: "What vegetarian dishes are available?"
        Query:
        sql
        SELECT name, price, active  
        FROM menu  
        WHERE category = 'Vegetarian' AND price > 0.00  
        LIMIT {{top_k}};  

    Booking Query -
        User: "Can you tell me my reservation details?"
        Query:
        sql
        SELECT id, time, guests  
        FROM bookings  
        WHERE email = '{email}'  
        LIMIT {{top_k}};

    General Query -
        User: "What are your opening hours?"
        Response: "Our opening hours are from 11:00 AM to 10:00 PM."

    Unauthorized Query -
        User: "How many orders were placed last month?"
        Response: "Sorry, I can't provide authorized information from the restaurant. You can ask about your bookings, {RESTAURANT_NAME} related services and other general informations."

    **Final Notes**
    Always ensure the generated query is precise, efficient, and safe to execute.
    If the user’s query is ambiguous or unclear, ask for clarification before proceeding.

    Maintain a polite and professional tone in all responses."""
            system_message = prompt_template.format(dialect="mysql", top_k=5)
            agent_executor = create_react_agent(llm, tools, prompt=system_message)

            final_answer = ""
            for step in agent_executor.stream(
                {"messages": [{"role": "user", "content": query}]},
                stream_mode="values",
            ):
                final_answer = step["messages"][-1].content.strip()
                replacements = {
                    "'": "''",
                    '"': '""',
                    "\\": "\\\\",
                }

            for old, new in replacements.items():
                final_answer = final_answer.replace(old, new)

            if any(
                phrase in final_answer
                for phrase in [
                    "I cannot retrieve",
                    "not enough information",
                    "I don''t have",
                    "I don''t know.",
                    "I don''t know",
                ]
            ):
                tavily_response = tavily_search(db_name=db_name, email=email, input=query)

                replacements = {
                    "'": "''",
                    '"': '""',
                    "\\": "\\\\",
                }

                for old, new in replacements.items():
                    tavily_response = tavily_response.replace(old, new)

                # Store the external response in memory instead of "I don't know"
                store_user_memory(
                    db=db, email=email, user_query=query, ai_response=tavily_response
                )
                return {"ai_response": tavily_response}

            # Store the valid AI-generated response in memory
            store_user_memory(db=db, email=email, user_query=query, ai_response=final_answer)

            return {"ai_response": final_answer}
        else:
            return {
                "ai_response": (
                    "Email authentication failed. Authentication is required to book a table or place an order."
                    "However, you are welcome to ask general inquiries."
                )
            }
    except Exception as error:
        return {
            "ai_response": f"There is an error occured.\nError: {error}",
        }


def book_table(
    db_name: str,
    name: str,
    phone: str,
    email: str,
    date: date,
    time: str,
    guests: str,
    message: str,
    created_at: str,
    updated_at: str,
):
    try:
        # Initialize database
        db = initialize_db(db_name=db_name)

        auth_query = f"SELECT * FROM `users` WHERE email = '{email}';"
        is_user_exists = db.run(auth_query)

        if is_user_exists:
            # Generate timestamps
            status = 0
            created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Convert date to string for SQL
            date_str = date.strftime("%Y-%m-%d")

            # Convert user input time string to 24-hour format
            time_str_24hr = convert_to_24hr(time)

            query = f"""
            INSERT INTO bookings (name, phone, email, date, time, guests, message, type, status, created_at, updated_at)
            VALUES ('{name}', '{phone}', '{email}', '{date_str}', '{time_str_24hr}', '{guests}', '{message}', '{1}', '{status}', '{created_at}', '{updated_at}')
            """
            db.run(query)

            return {
                "ai_response": "Your table has been booked successfully!",
            }
        else:
            return {
                "ai_response": (
                    "Email authentication failed. Authentication is required to book a table or place an order."
                    "However, you are welcome to ask general inquiries."
                )
            }
    except Exception as error:
        return {
            "ai_response": f"There is an error occured.\nError: {error}",
        }


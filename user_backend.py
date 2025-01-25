import os

from datetime import date, datetime, time
from utils import initialize_db
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits import create_sql_agent
from agents.tavily_search_agent import tavily_search_agent
from langchain_core.messages import SystemMessage

from dotenv import load_dotenv
load_dotenv()


def query_db_for_user(email: str, query: str):
    # Initialize database
    db = initialize_db(db_name="royce_balti")

    # Initialize the LLM model
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,  # Keep temperature low for accuracy
        max_retries=3,  # Increase retries for robustness
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    # Define the system prompt
    SQL_PREFIX = """You are an intelligent agent designed to interact with a SQL database and assist customers with their queries about orders, bookings, and restaurant menus. Your task is to construct accurate and syntactically correct SQLite queries to fetch the requested information. Follow these guidelines:

    Understand the Database Structure:
    Always start by inspecting the tables in the database to understand the available data. Never skip this step.

    Query Construction:
    Based on the user's question, create a precise SQLite query that retrieves only the relevant columns. Avoid selecting all columns (*) from a table unless explicitly instructed.

    Query Execution and Results:
    Limit the query to at most 5 results unless the user specifies otherwise.
    Order the results by a relevant column to present the most meaningful data.

    Error Handling:
    Double-check the query syntax before execution.
    If a query fails, revise it and try again.

    Read-Only Access:
    Only perform SELECT statements. Do not execute any data manipulation statements (INSERT, UPDATE, DELETE, DROP, etc.).

    Response Formation:
    Use the information returned by the database to provide clear and accurate answers to the user’s questions."""

    # Format the system message
    system_message = SystemMessage(content=SQL_PREFIX)

    # Check if the user exists in the database
    auth_query = f"""SELECT * FROM `users` WHERE email = "{email}";"""
    is_user_exists = db.run(auth_query)

    if is_user_exists:
        # Create the SQL agent with the system message
        agent_executor = create_sql_agent(
            llm=llm, db=db, verbose=True, handle_parsing_errors=True, message_modifier=system_message
        )

        try:
            # Enforce the email constraint for booking-related queries
            if "booking" in query.lower():
                query = f"{query.strip()} AND email = '{email}'"

            # Execute the query using the agent
            response = agent_executor.invoke({"input": query})

            # Check for fallback response
            if "I don't know" in response["output"]:
                response = tavily_search_agent(input=query)  # Use fallback agent for general queries
                return {"ai_response": response}
            else:
                return {"ai_response": response["output"]}
        except Exception as e:
            # Handle exceptions gracefully
            return {"ai_response": f"Error processing query: {str(e)}"}
    else:
        # Return an authentication error if the user is not found
        return {"ai_response": "Email authentication failed. Authentication is required to book a table or place an order. However, you are welcome to ask general inquiries."}
    

def book_table(
    name: str,
    phone: str,
    email: str,
    date: date,
    time: time,
    guests: str,
    message: str,
    booking_type: str,
    created_at: str,
    updated_at: str,
):
    # Generate timestamps
    status = "Pending"
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Convert date and time to strings for SQL
    date_str = date.strftime("%Y-%m-%d")
    time_str = time.strftime("%H:%M:%S")

    db = initialize_db(db_name="royce_balti")
    query = f"""
    INSERT INTO bookings (name, phone, email, date, time, guests, message, type, status, created_at, updated_at)
    VALUES ('{name}', '{phone}', '{email}', '{date_str}', '{time_str}', '{guests}', '{message}', '{booking_type}', '{status}', '{created_at}', '{updated_at}')
    """
    db.run(query)

    return {
        "ai_response": "Your table has been booked successfully with the following details:",
        "name": name,
        "phone": phone,
        "email": email,
        "date": date,
        "time": time,
        "guests": guests,
        "message": message,
        "booking_type": booking_type,
    }

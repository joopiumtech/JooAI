import os

from datetime import date, datetime, time
from typing import Any
from utils import initialize_db
from langchain_google_genai import ChatGoogleGenerativeAI
from agents.tavily_search_agent import tavily_search_agent
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from dotenv import load_dotenv
load_dotenv()


# Initialize database
db = initialize_db(db_name="roycebalti")

# Initialize the LLM model
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
    api_key=os.environ.get("OEPNAI_API_KEY")
)


def query_db_for_user(email: str, query: str):
    """
    Authenticates a user by email and processes their query to the database using an LLM-powered agent.

    Args:
        email (str): User's email address for authentication.
        query (str): User's query to be executed on the database.

    Returns:
        dict: A dictionary containing the AI-generated response.
    """

    def authenticate_user(email: str) -> bool:
        """Checks if the user exists in the database."""
        auth_query = f"SELECT * FROM `users` WHERE email = '{email}';"
        return bool(db.run(auth_query))

    def create_agent_executor() -> Any:
        """Creates and returns an agent executor configured for the database."""
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        tools = toolkit.get_tools()

        prompt_template = (
            """You are an agent designed to interact with a SQL database.
            Given an input question, create a syntactically correct {dialect} query to run, 
            then look at the results of the query and return the answer.
            Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
            You can order the results by a relevant column to return the most interesting examples in the database.
            You have only access to the following tables 'booking', 'coupon', 'orders', and 'menu'. 
            If the user asks questions about other tables, respond strictly with "Apologies, you are not authorized to access this information. Feel free to ask about bookings, orders, restaurant services, or general inquiries.".
            Never query for all columns from a table; only query for the relevant columns given the question.
            Double-check your query before executing it. If you encounter an error, rewrite the query and try again.
            Do NOT make any DML statements (INSERT, UPDATE, DELETE, DROP, etc.) to the database.
            Always start by examining the schema of the most relevant tables.
            """
        )
        system_message = prompt_template.format(dialect="mysql", top_k=5)

        return create_react_agent(llm, tools, prompt=system_message)

    def process_query(agent_executor, email: str, query: str) -> str:
        """Processes the user's query using the agent executor and streams results."""
        # Append email to queries involving 'booking'
        if "booking" in query:
            query = f"{query.strip()} for {email}"

        final_answer = ""
        for step in agent_executor.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
        ):
            final_answer = step["messages"][-1].content.strip()

        return final_answer

    def handle_fallback(query: str) -> str:
        """Handles fallback logic if the primary agent cannot fulfill the query."""
        return tavily_search_agent(input=query)

    # Authenticate the user
    if not authenticate_user(email):
        return {
            "ai_response": (
                "Email authentication failed. Authentication is required to book a table or place an order. "
                "However, you are welcome to ask general inquiries."
            )
        }

    # Create the agent executor
    agent_executor = create_agent_executor()

    # Process the query
    final_answer = process_query(agent_executor, email, query)

    # Check if fallback is needed
    if any(phrase in final_answer for phrase in ["I cannot retrieve", "not enough information", "I don't have"]):
        fallback_response = handle_fallback(query)
        return {"ai_response": fallback_response}

    return {"ai_response": final_answer}




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
    status = 0
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Convert date and time to strings for SQL
    date_str = date.strftime("%Y-%m-%d")
    time_str = time.strftime("%H:%M:%S")

    db = initialize_db(db_name="roycebalti")
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

import os
import ast

from datetime import date, datetime, time
from typing import Any
from utils import convert_to_24hr, initialize_db
from langchain_google_genai import ChatGoogleGenerativeAI
from agents.tavily_search_agent import tavily_search
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.memory import ConversationBufferMemory


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
    api_key=os.environ.get("OPENAI_API_KEY")
)


def get_user_memory(email: str):
    """Retrieve the last few interactions from MySQL memory."""
    query = f"""SELECT user_query, ai_response FROM user_memory WHERE email = '{email}' ORDER BY timestamp DESC LIMIT 3"""
    response = db.run(query)
    return response


def store_user_memory(email: str, user_query: str, ai_response: str):
    """Store user interactions in MySQL memory."""
    query = f"""INSERT INTO user_memory (email, user_query, ai_response) VALUES ('{email}', '{user_query.strip()}', '{ai_response}')"""
    db.run(query)


def query_db_for_user(email: str, query: str):
    """
    Authenticates a user by email and processes their query to the database using an LLM-powered agent.

    Args:
        email (str): User's email address for authentication.
        query (str): User's query to be executed on the database.

    Returns:
        dict: A dictionary containing the AI-generated response.
    """
    auth_query = f"SELECT * FROM `users` WHERE email = '{email}';"
    is_user_exists = db.run(auth_query)

    if is_user_exists:
        """Creates and returns an agent executor configured for the database."""
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        tools = toolkit.get_tools()

        # Retrieve memory context
        past_interactions = get_user_memory(email) or "[]"
        past_interactions = ast.literal_eval(past_interactions)

        # Check if past_interactions has any data
        if past_interactions:
            memory_context = "\n".join([f"user: {q}\nai_response: {r}" for q, r in past_interactions])
        else:
            memory_context = ""  # Empty memory context if no past interactions


        prompt_template = f"""You are an intelligent agent designed to interact with a SQL database for a restaurant chatbot.
        Given an chat history: {memory_context} and input question: {query}, generate a syntactically correct {{dialect}} query to retrieve the relevant information.
        You have only access to the following tables bookings, orders, menu, coupons, users.
        
        Guidelines:
        Always limit queries to at most {{top_k}} results unless the user specifies otherwise.
        Order results by relevance, such as popularity, price, or reservation time.
        Only query for necessary columns—never use SELECT *.
        Always check the database schema for the most relevant tables before constructing your query.
        If a query fails, refine it and try again.
        Never execute DML statements (INSERT, UPDATE, DELETE, DROP, etc.).
        If you don't know the answer. Strictly respond with "I don't know".
 
        Capabilities:
        Menu Queries: Retrieve dish details, prices, availability. (NOTE: Exclude price 0.00)
        Booking Queries: If query is related to booking. Only retrieve the booking details for user: {email}. If booking with {email} not exists. Strictly respond with "Unable to find any booking details associated with the email address {email}. Please double-check the information or contact us for further assistance."
        General Queries: Answer general knowledge questions.
        
        Exclude from response:
        Don't include any private restaurant details in the response (eg: Total sales details, Total orders). If user ask about it. Strictly respond with. "Sorry, I can't provide authorized informations from the restaurant. You can ask quries related about your bookings, orders, and other general informations."
        Dish recommendations, generic answers about the business but anything specific in our database is prohibited to the public. Eg: a customer can ask what is the most popular dish , but he cannot ask how many times it is ordered 
        
        Ensure the generated query is precise, efficient, and safe to execute."""
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


        if any(phrase in final_answer for phrase in ["I cannot retrieve", "not enough information", "I don''t have", "I don''t know.", "I don''t know"]):
            tavily_response = tavily_search(email=email, input=query)

            replacements = {
                "'": "''",
                '"': '""',
                "\\": "\\\\",
            }

            for old, new in replacements.items():
                tavily_response = tavily_response.replace(old, new)

            # Store the external response in memory instead of "I don't know"
            store_user_memory(email, query, tavily_response)
            return {"ai_response": tavily_response}
        
        # Store the valid AI-generated response in memory
        store_user_memory(email, query, final_answer)   

        return {"ai_response": final_answer}

    else:
        return {
                "ai_response": (
                    "Email authentication failed. Authentication is required to book a table or place an order."
                    "However, you are welcome to ask general inquiries."
                )
            }



def book_table(
    name: str,
    phone: str,
    email: str,
    date: date,
    time: str,
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

    # Convert date to string for SQL
    date_str = date.strftime("%Y-%m-%d")
    
    # Convert user input time string to 24-hour format
    time_str_24hr = convert_to_24hr(time)

    query = f"""
    INSERT INTO bookings (name, phone, email, date, time, guests, message, type, status, created_at, updated_at)
    VALUES ('{name}', '{phone}', '{email}', '{date_str}', '{time_str_24hr}', '{guests}', '{message}', '{booking_type}', '{status}', '{created_at}', '{updated_at}')
    """
    db.run(query)

    return {
        "ai_response": "Your table has been booked successfully with the following details:",
        "name": name,
        "phone": phone,
        "email": email,
        "date": date,
        "time": time_str_24hr,
        "guests": guests,
        "message": message,
        "booking_type": booking_type,
    }

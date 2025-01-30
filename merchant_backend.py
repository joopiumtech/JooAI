import ast
import os
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from agents.tavily_search_agent import tavily_search
from utils import initialize_db
from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain import hub
from langchain_openai import ChatOpenAI

# Load environment variables
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


def get_merchant_memory(email: str):
    """Retrieve the last few interactions from MySQL memory."""
    query = f"""SELECT merchant_query, ai_response FROM merchant_memory WHERE email = '{email}' ORDER BY timestamp DESC LIMIT 5"""
    response = db.run(query)
    return response


def store_merchant_memory(email: str, merchant_query: str, ai_response: str):
    """Store user interactions in MySQL memory."""
    query = f"""INSERT INTO merchant_memory (email, merchant_query, ai_response) VALUES ('{email}', '{merchant_query.strip()}', '{ai_response}')"""
    db.run(query)


def query_db_for_merchant(email: str, query: str):
    """
    Authenticates a merchant by email and processes their query to the database using an LLM-powered agent.

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
        past_interactions = get_merchant_memory(email) or "[]"
        past_interactions = ast.literal_eval(past_interactions)

        # Check if past_interactions has any data
        if past_interactions:
            memory_context = "\n".join([f"user: {q}\nai_response: {r}" for q, r in past_interactions])
        else:
            memory_context = ""  # Empty memory context if no past interactions


        prompt_template = f"""You are an intelligent agent designed to interact with a SQL database for a restaurant chatbot.
        Given an chat history: {memory_context} and input question: {query}, generate a syntactically correct {{dialect}} query to retrieve the relevant information. Based on
        
        Guidelines:
        Always limit queries to at most {{top_k}} results unless the user specifies otherwise.
        Order results by relevance, such as popularity, price, or reservation time.
        Only query for necessary columns—never use SELECT *.
        Always check the database schema for the most relevant tables before constructing your query.
        If a query fails, refine it and try again.
        Never execute DML statements (INSERT, UPDATE, DELETE, DROP, etc.).
        If you don't know the answer. Strictly respond with "I don't know".
        
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
            store_merchant_memory(email, query, tavily_response)
            return {"ai_response": tavily_response}
        
        # Store the valid AI-generated response in memory
        store_merchant_memory(email, query, final_answer)   

        return {"ai_response": final_answer}

    else:
        return {
                "ai_response": (
                    "Email authentication failed. Authentication is required to book a table or place an order."
                    "However, you are welcome to ask general inquiries."
                )
            }


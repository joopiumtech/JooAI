import ast
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from agents.tavily_search_agent import tavily_search
from utils import fetch_restaurant_name, initialize_db, store_merchant_memory, get_merchant_memory
from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain import hub
from langchain_openai import ChatOpenAI

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


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
    api_key=os.environ.get("OPENAI_API_KEY"),
    streaming=True
)


def query_db_for_merchant(db_name: str, email: str, query: str):
    """
    Authenticates a merchant by email and processes their query to the database using an LLM-powered agent.

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
            chat_history = get_merchant_memory(db=db, email=email) or "[]"
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
            prompt_template = f"""Restaurnat Name: {RESTAURANT_NAME}
You are an intelligent agent designed to interact with a SQL database for a restaurant merchant chatbot.
Given an chat history:\n{memory_context}\nand input question: {query}.\ngenerate a syntactically correct {{dialect}} query to retrieve the relevant information.

Guidelines:
Always limit queries to at most {{top_k}} results unless the merchant specifies otherwise.
Order results by relevance, such as popularity, price, or reservation time.
Only query for necessary columns—never use SELECT *.
Always check the database schema for the most relevant tables before constructing your query.
If a query fails, refine it and try again.
Never execute DML statements (INSERT, UPDATE, DELETE, DROP, etc.).
If you don't know the answer. Strictly repond with "I don't know".

Ensure the generated query is precise, efficient, and safe to execute."""
            system_message = prompt_template.format(dialect="mysql", top_k=5)
            agent_executor = create_react_agent(llm, tools, prompt=system_message)

            response = agent_executor.invoke({"messages": [{"role": "user", "content": query}]})
            final_answer = response["messages"][-1].content.strip()

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
                store_merchant_memory(
                    db=db, email=email, merchant_query=query, ai_response=tavily_response
                )
                return {"ai_response": tavily_response}

            # Store the valid AI-generated response in memory
            store_merchant_memory(db=db, email=email, merchant_query=query, ai_response=final_answer)

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
            "ai_response": f"There is an error occured.\nError: {error}"
        }

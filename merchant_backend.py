import os
import ast

from agents.tavily_search_agent import tavily_search
from utils import fetch_restaurant_name, initialize_db, store_merchant_memory, get_merchant_memory, verify_password
from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status

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


# Secret Key for JWT
SECRET_KEY = os.environ.get("JWT_SECRET")
ALGORITHM = os.environ.get("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth")

# Hashing password
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)



def auth_test(email: str, password: str):
    try:
        db = initialize_db()
        
        email_query = f"""SELECT COUNT(*) > 0 AS user_exists FROM users WHERE email = '{email}';"""
        email_address = ast.literal_eval(db.run(email_query))

        if email_address[0][0] == 0:
            return {"is_authenticated": False,
                    "auth_message": "Email address not found. Please check your email address and try again."} 
        else:
            password_get_query = f"""SELECT password FROM users where email='{email}'"""
            hashed_password = ast.literal_eval(db.run(password_get_query))
            is_password_correct = verify_password(password, hashed_password[0][0])


            if is_password_correct:
                # Generate JWT token
                access_token = create_access_token(data={"sub": email})
                return {"is_authenticated": True,
                        "auth_message": "Authenticated successfully",
                        "access_token": access_token}
            else:
                return {"is_authenticated": False,
                        "auth_message": "Wrong password. Please check your password and try again."}
    except Exception as error:
        return {
            "is_authenticated": False,
            "auth_message": f"There is an error occured.\nError: {error}"
        }
    



def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Validates JWT token and returns user email.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")

        if email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return email
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )



def query_db_for_merchant(query: str):
    """
    Authenticates a merchant by email and processes their query to the database using an LLM-powered agent.

    Args:
        query (str): User's query to be executed on the database.

    Returns:
        dict: A dictionary containing the AI-generated response.
    """
    try:
        # Initialize database
        db = initialize_db()
        email = os.environ.get("ADMIN_EMAIL")


        """Creates and returns an agent executor configured for the database."""
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        tools = toolkit.get_tools()
        
        # Retrieve memory context
        chat_history = get_merchant_memory(email=email) or "[]"
        chat_history = ast.literal_eval(chat_history)

        # Check if chat_history has any data
        if chat_history:
            memory_context = "\n".join(
                [f"user: {q}\nai_response: {r}" for q, r in chat_history]
            )
        else:
            memory_context = ""  # Empty memory context if no past interactions

        # HYPERPARAMETERS
        RESTAURANT_NAME = fetch_restaurant_name()
        prompt_template = f"""Restaurnat Name: {RESTAURANT_NAME}
        You are an AI assistant designed to interact with {RESTAURANT_NAME}'s SQL database to answer queries related to the restaurant's operations, based on the chat history: {memory_context} and input question: {query}.

        Guidelines for Query Execution:
        Understanding the Database:
        - Before generating any query, first retrieve and examine the available tables to understand what data can be accessed.
        - Identify the most relevant tables and check their schema before constructing your query.

        Constructing SQL Queries:
        - Generate only syntactically correct {{dialect}} queries.
        - Focus only on relevant columns instead of selecting all columns from a table.
        - Unless the user specifies a particular number of results, limit queries to {{top_k}} results for efficiency.
        - When applicable, order results by a relevant column to provide the most insightful answers.

        Execution & Error Handling:
        - Always double-check your query before execution.
        - If an error occurs, refine the query and retry instead of returning incorrect results.
        - For queries related to bookings, retrieve only the most up-to-date information from the bookings table. If no data is available, respond strictly with: "Currently, there are no booking records available." 
        - If you can answer it directly, do so.
        - If you don't know the answer, strictly respond with "I don't know". Don't try to create an answer from the data.

        Restrictions:
        - Do NOT execute any DML (INSERT, UPDATE, DELETE, DROP, etc.) operations—your role is strictly read-only.
        - Only use the tools provided to interact with the database and rely solely on the returned data to construct responses.
        
        Your goal is to provide accurate, concise, and insightful answers based on the restaurant's data."""
        
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
            tavily_response = tavily_search(input=query)

            replacements = {
                "'": "''",
                '"': '""',
                "\\": "\\\\",
            }

            for old, new in replacements.items():
                tavily_response = tavily_response.replace(old, new)

            # Store the external response in memory instead of "I don't know"
            store_merchant_memory(email=email, merchant_query=query, ai_response=tavily_response)
            return {"ai_response": tavily_response}

        # Store the valid AI-generated response in memory
        store_merchant_memory(email=email, merchant_query=query, ai_response=final_answer)

        return {"ai_response": final_answer}
       
    except Exception as error:
        return {
            "ai_response": f"There is an error occured.\nError: {error}"
        }



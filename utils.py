import os
import ast
import bcrypt


from langchain_community.utilities import SQLDatabase

from dotenv import load_dotenv
load_dotenv(override=True)


def initialize_db():
    # Initialize database connection with environment variables
    db_user = os.environ.get("DB_USER")
    db_host = os.environ.get("DB_HOST")
    db_pass = os.environ.get("DB_PASS")
    db_name = os.environ.get("DB_NAME")
    # Construct database URI and initialize connection
    database_uri = f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}/{db_name}"
    db = SQLDatabase.from_uri(database_uri)

    return db


# Merchant Memory
def get_merchant_memory(email: str):
    db = initialize_db()
    """Retrieve the last few interactions from MySQL memory."""
    query = f"""SELECT merchant_query, ai_response FROM merchant_memory WHERE email = '{email}' ORDER BY timestamp"""
    response = db.run(query)
    return response


def store_merchant_memory(email: str, merchant_query: str, ai_response: str):
    db = initialize_db()
    """Store user interactions in MySQL memory."""
    query = f"""INSERT INTO merchant_memory (email, merchant_query, ai_response) VALUES ('{email}', '{merchant_query.strip()}', '{ai_response}')"""
    db.run(query)


def fetch_restaurant_name():
    db = initialize_db()

    sql_query = f"""SELECT bannertitle FROM homepages"""
    response = db.run(sql_query)
    modified_res = ast.literal_eval(response)
    return modified_res[0]


def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))


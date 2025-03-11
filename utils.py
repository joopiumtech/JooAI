import os
import ast
import bcrypt
import redis
import json

from langchain_community.utilities import SQLDatabase

from dotenv import load_dotenv
load_dotenv(override=True)


# ----------------------------------------------------------------------------
# DB Utils
# ----------------------------------------------------------------------------

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


def fetch_restaurant_name():
    db = initialize_db()

    sql_query = f"""SELECT name FROM settings"""
    response = db.run(sql_query)
    modified_res = ast.literal_eval(response)
    return modified_res[0][0]


def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))



# ----------------------------------------------------------------------------
# Memory Utils
# ----------------------------------------------------------------------------

# DB Memory Setup
# def get_merchant_memory(email: str):
#     db = initialize_db()
#     """Retrieve the last few interactions from MySQL memory."""
#     query = f"""SELECT merchant_query, ai_response FROM merchant_memory WHERE email = '{email}' ORDER BY timestamp"""
#     response = db.run(query)
#     return response


# def store_merchant_memory(email: str, merchant_query: str, ai_response: str):
#     db = initialize_db()
#     """Store user interactions in MySQL memory."""
#     query = f"""INSERT INTO merchant_memory (email, merchant_query, ai_response) VALUES ('{email}', '{merchant_query.strip()}', '{ai_response}')"""
#     db.run(query)


def initialize_redis():
    # Connect to Redis (default: localhost:6379)
    r = redis.Redis(
        host=os.environ.get("REDIS_HOST"),
        port=os.environ.get("REDIS_PORT"),
        decode_responses=True,
        username=os.environ.get("REDIS_USERNAME"),
        password=os.environ.get("REDIS_PASS"),
    )
    return r


def insert_data_to_redis(email: str, query: str, ai_response: str):
    r = initialize_redis()
    key = f"{email}_memory"

    new_data = {"query": query, "ai_response": ai_response}

    # Push new JSON object to Redis list
    r.rpush(key, json.dumps(new_data))



def retrieve_data_from_redis(email: str):
    r = initialize_redis()
    key = f"{email}_memory"
    
    # Retrieve all elements in the list
    return [json.loads(item) for item in r.lrange(key, 0, -1)]


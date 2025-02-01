import ast
import os
import re

from langchain_community.utilities import SQLDatabase

from dotenv import load_dotenv

load_dotenv()


def initialize_db(db_name):
    # Initialize database connection with environment variables
    db_user = os.environ.get("DB_USER")
    db_host = os.environ.get("DB_HOST")
    db_pass = os.environ.get("DB_PASS")

    # Construct database URI and initialize connection
    database_uri = f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}/{db_name}"
    db = SQLDatabase.from_uri(database_uri)

    return db


def convert_to_24hr(time_str: str) -> str:
    # Match time in format like '7pm', '7am', '12pm', '12am', etc.
    match = re.match(r"(\d{1,2})(am|pm)", time_str.strip(), re.IGNORECASE)

    if match:
        hour = int(match.group(1))
        period = match.group(2).lower()

        # Adjust hour for 12-hour to 24-hour conversion
        if period == "am":
            if hour == 12:
                hour = 0  # Midnight case
        elif period == "pm":
            if hour != 12:
                hour += 12  # PM case, add 12 for hours 1-11

        # Format the time as HH:MM:00
        return f"{hour:02}:00:00"

    raise ValueError("Invalid time format. Please use 'am' or 'pm'.")


def get_memory(db, email: str):
    """Retrieve the last few interactions from MySQL memory."""
    query = f"""SELECT user_query, ai_response FROM user_memory WHERE email = '{email}' ORDER BY timestamp DESC LIMIT 3"""
    response = db.run(query)
    return response


def store_memory(db, email: str, user_query: str, ai_response: str):
    """Store user interactions in MySQL memory."""
    query = f"""INSERT INTO user_memory (email, user_query, ai_response) VALUES ('{email}', '{user_query.strip()}', '{ai_response}')"""
    db.run(query)


def fetch_restaurant_name(db):
    sql_query = f"""SELECT bannertitle FROM homepages"""
    response = db.run(sql_query)
    modified_res = ast.literal_eval(response)
    return modified_res[0]

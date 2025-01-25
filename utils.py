import os

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
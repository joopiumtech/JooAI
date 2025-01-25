import os

from fastapi import HTTPException

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits import create_sql_agent
from agents.tavily_search_agent import tavily_search_agent
from utils import initialize_db

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


def query_db_for_merchant(query: str):
    """
    Query the database for merchant-related information using a SQL agent and LLM.

    Args:
        query (str): The merchant's query string.

    Returns:
        dict: A dictionary containing the original query and the AI-generated response.

    Raises:
        HTTPException: If an error occurs during query execution.
    """
    db = initialize_db(db_name="royce_balti")

    # Initialize the LLM model
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,  # Keep temperature low for accuracy
        max_retries=3,  # Increase retries for robustness
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    # Define constants
    RESTAURANT_NAME = "Royce Balti"

    # Craft a dynamic prompt with the query injected
    prompt_template = f"""
    Restaurant Name: {RESTAURANT_NAME}

    You are an AI merchant chatbot integrated with {RESTAURANT_NAME}'s SQL database. Your job is to provide clear, accurate, and actionable responses to the merchant query: {query}.

    Response Guidelines:

    Context-Driven Responses:
    Use only the data available in the SQL database (menu, orders, inventory, bookings, etc.).
    Always prioritize accuracy and relevance.

    Clarity and Conciseness:
    Provide precise answers without unnecessary details.

    Data Privacy:
    Do not disclose sensitive internal identifiers unless explicitly requested.

    Formatting:
    Use clear numerical formats (e.g., "Total sales: $5,432").
    Present data in a table format when listing multiple items (e.g., menus, reports).

    Handling Specific Queries:
    For price-related queries, retrieve the highest value from the "price" column of the menu table.

    Tone:
    Maintain a professional yet approachable tone suitable for a restaurant setting.

    Examples:
    Query: "What are the top-selling dishes this week?"
    Response: "The top-selling dishes this week are Chicken Tikka Masala, Lamb Biryani, and Paneer Butter Masala."

    Query: "How many orders were completed yesterday?"
    Response: "A total of 56 orders were completed yesterday."

    Query: "What was the total revenue for last week?"
    Response: "The total revenue for last week was $7,890."

    Query: "How many new customers did we have this month?"
    Response: "You had 25 new customers this month."
    """

    # Format the system message
    system_message = prompt_template.format(dialect="mysql", top_k=5)

    # Create a SQL agent with verbose output for debugging
    agent_executor = create_sql_agent(
        llm=llm, db=db, verbose=True, handle_parsing_errors=True, state_modifier=system_message
    )

    # Execute the prompt and handle parsing errors gracefully
    try:
        response = agent_executor.invoke(query)
        if "I don't know" in response["output"]:
            fallback_response = tavily_search_agent(input=query)
            return {
                "ai_response": fallback_response,
            }
        else:
            return {
                "ai_response": response["output"],
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
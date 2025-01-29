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
    api_key=os.environ.get("OEPNAI_API_KEY")
)


def query_db_for_merchant(query: str):
    """
    Processes a query for a merchant using an LLM-powered agent.

    Args:
        query (str): The user's query to execute on the database.

    Returns:
        dict: A dictionary containing the AI-generated response.
    """

    def create_agent_executor() -> Any:
        """Creates and returns an agent executor configured for the database."""
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        tools = toolkit.get_tools()

        # Pull system prompt template and configure it
        prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
        system_message = prompt_template.format(dialect="mysql", top_k=5)

        return create_react_agent(llm, tools, prompt=system_message)

    def process_query(agent_executor, query: str) -> str:
        """Processes the user's query using the agent executor and streams results."""
        final_answer = ""
        for step in agent_executor.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
        ):
            final_answer = step["messages"][-1].content.strip()

        return final_answer

    def handle_fallback(query: str) -> str:
        """Handles fallback logic if the primary agent cannot fulfill the query."""
        return tavily_search(input=query)

    # Create the agent executor
    agent_executor = create_agent_executor()

    # Process the query
    final_answer = process_query(agent_executor, query)

    # Check if fallback is needed
    if any(phrase in final_answer for phrase in ["I cannot retrieve", "not enough information", "I don't have"]):
        fallback_response = handle_fallback(query)
        return {"ai_response": fallback_response}

    return {"ai_response": final_answer}


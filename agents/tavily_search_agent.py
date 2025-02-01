import ast
import os


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from tools.tool import search_on_tavily
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain import hub

from dotenv import load_dotenv

from utils import initialize_db

load_dotenv()

# Initialize database
db = initialize_db(db_name="roycebalti")


def get_user_memory(email: str):
    """Retrieve the last few interactions from MySQL memory."""
    query = f"""SELECT user_query, ai_response FROM user_memory WHERE email = '{email}' ORDER BY timestamp DESC LIMIT 5"""
    response = db.run(query)
    return response


def store_user_memory(email: str, user_query: str, ai_response: str):
    """Store user interactions in MySQL memory."""
    query = f"""INSERT INTO user_memory (email, user_query, ai_response) VALUES ('{email}', '{user_query.strip()}', '{ai_response}')"""
    db.run(query)


def tavily_search(email: str, input: str):
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_retries=1,
        api_key=os.environ.get("OEPNAI_API_KEY"),
    )

    # Retrieve memory context
    past_interactions = get_user_memory(email) or "[]"
    past_interactions = ast.literal_eval(past_interactions)

    # Check if past_interactions has any data
    if past_interactions:
        memory_context = "\n".join(
            [f"user: {q}\nai_response: {r}" for q, r in past_interactions]
        )
    else:
        memory_context = ""  # Empty memory context if no past interactions

    template = f"""
    Based on the chat history: {memory_context} and user input: "{input}", if you can answer it directly, do so. If you cannot find the answer immediately, 
    you should either search for updated information on the web or request more clarification from the user. 
    Please make sure the action step is clear and actionable.
    """

    prompt_template = PromptTemplate(template=template, input_variables=["input"])

    tools_for_agent = [
        Tool(
            name=f"Crawl Google for {input}",
            func=search_on_tavily,
            description="useful for get the updated information",
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools_for_agent, verbose=True, handle_parsing_errors=True
    )

    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(input=input)}
    )
    response = result["output"]
    return response

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
from utils import get_merchant_memory, get_user_memory, initialize_db

from dotenv import load_dotenv
load_dotenv()



def tavily_search(input: str):
    # Initialize database
    db = initialize_db()
    email = os.environ.get("ADMIN_EMAIL")

    # Initialize LLM model
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_retries=1,
        api_key=os.environ.get("OEPNAI_API_KEY"),
    )
    

    # Retrieve merchant memory context
    chat_history = get_merchant_memory(email=email) or "[]"
    chat_history = ast.literal_eval(chat_history)
    
    # Check if chat_history has any data
    if chat_history:
        memory_context = "\n".join(
            [f"user: {q}\nai_response: {r}" for q, r in chat_history]
        )
    else:
        memory_context = ""  # Empty memory context if no past interactions

    template = f"""Based on the chat history: {memory_context} and user input: {input} you should search for updated information on the web. Please make sure the information is clear and relevant."""

    prompt_template = PromptTemplate(template=template, input_variables=["input"])

    tools_for_agent = [
        Tool(
            name=f"Crawl Google for updated information",
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





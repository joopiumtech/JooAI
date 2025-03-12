import os
import ast


from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from tools.tool import search_on_tavily
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain import hub
from utils import get_merchant_memory

from dotenv import load_dotenv
load_dotenv()



def tavily_search(input: str):
    email = os.environ.get("ADMIN_EMAIL")

    # Initialize LLM model
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_retries=1,
        api_key=os.environ.get("OEPNAI_API_KEY"),
    )
    

    # Retrieve memory context
    chat_history = get_merchant_memory(email=email) or "[]"
    chat_history = ast.literal_eval(chat_history)

    # Process chat history
    memory_context = "\n".join(
        [f"merchant_query: {merchant_query}\nai_response: {ai_response}\n" for merchant_query, ai_response in chat_history]
    )

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
        agent=agent, tools=tools_for_agent, handle_parsing_errors=True
    )

    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(input=input)}
    )
    response = result["output"]
    return response





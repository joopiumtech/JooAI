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
load_dotenv()


def tavily_search_agent(input: str):
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_retries=1,
        api_key=os.environ.get("OEPNAI_API_KEY")
    )

    template = f"""
    Based on the user input: "{input}", if you can answer it directly, do so. If you cannot find the answer immediately, 
    you should either search for updated information on the web or request more clarification from the user. 
    Please make sure the action step is clear and actionable.
    """

    prompt_template = PromptTemplate(
        template=template, input_variables=["input"]
    )

    tools_for_agent = [
        Tool(
            name=f"Crawl Google for {input}",
            func=search_on_tavily,
            description="useful for get the updated information",
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True, handle_parsing_errors=True)

    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(input=input)}
    )
    response = result["output"]
    return response

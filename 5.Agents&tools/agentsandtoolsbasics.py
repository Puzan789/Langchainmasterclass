from dotenv import load_dotenv
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_react_agent
)
import os
from langchain_core.tools import Tool 
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()


os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
def get_current_time(*args,**kwargs):
    import datetime
    now=datetime.datetime.now()
    return now.strftime('%Y-%m-%d')

tools=[
    Tool(
        name="time",
        func=get_current_time,
        description="useful for when you need current time "
    )
]
prompt=hub.pull("hwchase17/react") 
llm=ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

agent=create_react_agent(
    tools=tools,
    llm=llm,
    prompt=prompt,
    stop_sequence=True
)

agent_executor=AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True
)

response=agent_executor.invoke({"input":"What time is it?"})
print(response['answer'])
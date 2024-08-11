from dotenv import load_dotenv
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
    create_structured_chat_agent,
)
import os
from langchain_core.tools import Tool 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage

load_dotenv()


os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
def get_current_time(*args,**kwargs):
    import datetime
    now=datetime.datetime.now()
    return now.strftime('%I:%M %p')

def search_wikipedia(query):
    from wikipedia import summary
    try:
        return summary(query,sentences=2)
    except:
        return "I couldn't find any information on that"
tools=[
    Tool(
        name="time",
        func=get_current_time,
        description="useful for when you need current time "
    ),
    Tool(
        name="wikipedia",
        func=search_wikipedia,
        description="This tool allows you to search Wikipedia for information"
    ),
]
prompt=hub.pull("hwchase17/structured-chat-agent") 
llm=ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

memory=ConversationBufferMemory(memory_key="chat_history",return_messages=True)
agent=create_structured_chat_agent(
    tools=tools,
    llm=llm,
    prompt=prompt,
    stop_sequence=True
)

agent_executor=AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
)

initial_message="you are an AI assistant that can provide helpful answer using avaialble tools.if ypu are unable to answer it just say i cant"
memory.chat_memory.add_message(HumanMessage(content=initial_message))

while True:
    user_input=input("User: ")
    if user_input.lower() == "exit":
        break
    
    memory.chat_memory.add_message(AIMessage(content=user_input))
    response=agent_executor.invoke({"input": user_input})
    print(f"AI: {response['output']}")
    memory.chat_memory.add_message(HumanMessage(content=response["output"]))
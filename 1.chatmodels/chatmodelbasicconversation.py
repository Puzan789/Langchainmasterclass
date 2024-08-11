from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
import os
load_dotenv()
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

#create a chatopenaimodel
llm=GoogleGenerativeAI(model="gemini-1.5-pro-latest")


## messages\
messages=[
    SystemMessage(content="solve the following math problems."),#The system message is used to communicate instructions or provide context to the model at the beginning of a conversation
    HumanMessage(content="1. 5+3"),
    HumanMessage(content="2. 4*2"),
    HumanMessage(content="3. 10/2"),
    HumanMessage(content="4. 100-50"),
    HumanMessage(content="5. 5*5*5"),
]
#define the prompt

result=llm.invoke(messages)
print(result)

messages=[
    SystemMessage(content="Solve the following Math problems"),
    HumanMessage(content="6*6"),
    AIMessage(content="its 36"),
    HumanMessage(content="what happens if it divided by5?"),


]
print(llm.invoke(messages))
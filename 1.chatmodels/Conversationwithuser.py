from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage,HumanMessage,SystemMessage
import os
load_dotenv()
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

#create a chatopenaimodel
llm=ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

chat_history=[] #using a list to store  messages

system_messages=SystemMessage(content="You are a helpful assistant")
chat_history.append(system_messages)


while True:
    query=input("You:")
    if query.lower() =="exit":
        break
    chat_history.append(HumanMessage(content=query))
    result=llm.invoke(chat_history)      

    #Get ai response 
    response=result.content
    chat_history.append(response)
    print(f"Chatbot:{response}")

print("________________________________")
print(chat_history)
        

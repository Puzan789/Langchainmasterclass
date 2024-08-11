from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
import os
load_dotenv()
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

#create a chatopenaimodel
llm=GoogleGenerativeAI(model="gemini-1.5-pro-latest")

#define the prompt
text="how to be respectful explain in one line."
result=llm.invoke(text)
print(result)
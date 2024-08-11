from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Load environment variables from .env file
load_dotenv()
os.environ["GOOGLE_API_KEY"] =os.getenv("GOOGLE_API_KEY")

llm=ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

# Define the ChatPromptTemplate
messages=[
    ('system',"you are a very much funny comedian who make people laugh until they cry on the {topic} in indian street."),
    ("human","tell me {count} jokes")
]

prompt=ChatPromptTemplate.from_messages(messages)
prompt=prompt.invoke({"topic":"british", "count":"3"})

result=llm.invoke(prompt)
print(result.content)


from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import os 
load_dotenv()
os.environ['GOOGLE_API_KEY'] =os.getenv('GOOGLE_API_KEY')

model=ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

messages=[
    ('system',"you are a very much funny comedian who make people laugh until they cry on the {topic} in street."),
    ("human","tell me {count} jokes")
]

prompt = ChatPromptTemplate.from_messages(messages)
chain=prompt|model|StrOutputParser() # yo strutputparser launubhaneko .content lagaunu jastai ho
result=chain.invoke({"topic":"pig", "count":3})
print(result)
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableLambda,RunnableSequence
import os 
load_dotenv()
os.environ['GOOGLE_API_KEY'] =os.getenv('GOOGLE_API_KEY')

model=ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

messages=[
    ('system',"you are a very much funny comedian who make people laugh until they cry on the {topic} in street."),
    ("human","tell me {count} jokes")
]
promt=ChatPromptTemplate.from_messages(messages)

format_prompt=RunnableLambda(lambda x : x.upper(),) 
count_words=RunnableLambda(lambda x : f"Word count :{len(x.split())}\n{x}")


chain=promt | model | StrOutputParser() | format_prompt | count_words
response=chain.invoke({"topic":"Indian astronaut","count":3})
print(response)
# see branching and extended also
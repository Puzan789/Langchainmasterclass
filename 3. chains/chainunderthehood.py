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

format_prompt=RunnableLambda(lambda x : promt.format(**x)) 
print(format_prompt)
invoke_model=RunnableLambda(lambda x :model.invoke(x))
parse_output=RunnableLambda(lambda x : x.content)

chain=RunnableSequence(first=format_prompt,middle=[invoke_model],last=parse_output)
response=chain.invoke({"topic":"lawyers","count":3})
print(response)


from langchain.prompts import ChatPromptTemplate 
from langchain_core.messages import HumanMessage

template="tell me a jokes about{input}."

prompt_template = ChatPromptTemplate.from_template(template)

prompt=prompt_template.invoke({"input": "roads"})

## Prompt with multiple placeholders 

template_m='''you are a helpful assistant
Human : Tell me {adj} jokes about a {input}.
Assistant:'''
prompt_multiple=ChatPromptTemplate.from_template(template_m)
prompt=prompt_multiple.invoke({"adj": "funny","input": "cats"})
print(prompt)


## prompts with System and human messages
messages=[
    ("system", "you are a funny comedian who tells jokes about the {topic}"),# if ypu want to replace values you havw to use tuples
     ("human", "tell me {counts} jokes.")
]
prompt_template=ChatPromptTemplate.from_messages(messages)
prompt=prompt_template.invoke({"topic":"Hungers","counts":"3"})

print(prompt)
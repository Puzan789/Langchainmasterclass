import os 
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor,create_react_agent
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage,HumanMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir,"db")
persistent_directory = os.path.join(db_dir,"chroma_db_withmetada")

# check if the Chroma Vector store already exists 
if os.path.exists(persistent_directory):
    print("Loading existing vector_store...")
    db=Chroma(persist_directory=persistent_directory,embedding_function=None)
else:
    raise FileNotFoundError(f"The directory {persistent_directory} does not exist")

embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")

db=Chroma(persist_directory=persistent_directory,embedding_function=embeddings)


retriever=db.as_retriever()

llm=ChatGoogleGenerativeAI(model="gemini-pro-1.5-latest")

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


history_Aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)

# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
# create a chain 
question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
rag_chain=create_retrieval_chain(history_Aware_retriever,question_answer_chain)

react_docstore_prompt=hub.pull("hwchase/react")

tools=[
    Tool(
        name=" Answer Question",
        func=lambda input ,**kwargs:rag_chain.invoke(
            {"input":input,"chat_history":kwargs.get("chat_history",[])}
        ),
        description="Ueful for when you need toanswer questions about the context",
    )
]
# Create the ReAct Agent with document store retriever
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_docstore_prompt,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, handle_parsing_errors=True
)

chat_history = []
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    response = agent_executor.invoke({"input": query, "chat_history": chat_history})
    print(f"AI: {response['output']}")

    # Update history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response["output"]))

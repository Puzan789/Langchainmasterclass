
import os 
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Define the directory containing the text files and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

print(f"Persistent directory: {persistent_directory}")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

loader=PyPDFDirectoryLoader("./pdf")
docs=loader.load()




text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=20)
texts = text_splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db=Chroma.from_documents(texts,embeddings,persist_directory=persistent_directory)

user_query = "what is embedding?"

retriever=db.as_retriever()

#
relevant_docs=retriever.invoke(user_query)

for i, doc in enumerate(relevant_docs, 1):
    print(f"Result {i}:")
    print(doc.page_content)
    print("--------------------------------------------------------------------------------------------------")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only. Please provide the most accurate response based on the question. 
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

llm=ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
# Create a chain of chains
document_chains=create_stuff_documents_chain(llm,prompt)

# Create a retrieval chain
retrieval_chain=create_retrieval_chain(retriever,document_chains)
response=retrieval_chain.invoke({"input":"what is embedding there?"})
print(response['answer'])

# for we we use webbaseloader

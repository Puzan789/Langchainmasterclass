import os 
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
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

retriever=db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k":3,"score_threshold": 0.1}  # Adjust the threshold as needed
)
#there are max marginal Relevance
retrievers = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,            # Number of results to retrieve
        "lambda_mult": 0.5 # Trade-off parameter between relevance and diversity
    }
)
#
relevant_docs=retriever.invoke(user_query)
relevant_docsd=retrievers.invoke(user_query)

for i, doc in enumerate(relevant_docs, 1):
    print(f"Result {i}:")
    print(doc.page_content)
    print("--------------------------------------------------------------------------------------------------")

print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Results based on Max Marginal Relevance (MMR) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
for i, doc in enumerate(relevant_docsd, 1):
    print(f"Result {i}:")
    print(doc.page_content)
    print("------------------")
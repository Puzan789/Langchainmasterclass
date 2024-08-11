import os 
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

loader=PyPDFDirectoryLoader("./pdf")
docs=loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=20)
texts = text_splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db=Chroma.from_documents(texts,embeddings)

user_query = "what is embedding?"

retriever=db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k":1,"score_threshold": 0.9}  # Adjust the threshold as needed
)
relevant_docs=retriever.invoke(user_query)

print(">><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<><><><<><><><><>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
for i , doc in enumerate(relevant_docs,1):
    print(f"Result {i}:")
    print(doc.page_content)
    print("--------------------------------------------------------------------------------------------------")  
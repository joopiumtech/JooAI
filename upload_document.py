import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv
load_dotenv(override=True)

embedding_model = OpenAIEmbeddings(
  model="text-embedding-3-large",
  api_key=os.environ.get('OPENAI_API_KEY')
)


def upload_pdf(pdf_path: str):
  '''
  This function is used to upload a document to the Pinecone vector database.
  
  Args:
    pdf_path: The path to the document to be uploaded.
  '''
  
  # Load the document
  loader = PyPDFLoader(pdf_path)
  texts = loader.load()

  # Splitting text into text_chunks
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    add_start_index=True
  )

  text_chunks = text_splitter.split_documents(texts)

  # Pinecone setup (Ingesting data into pinecone vector database)
  print("Uploading PDF to pinecone. Please wait...")
  PineconeVectorStore.from_documents(
    text_chunks,
    embedding_model,
    index_name=os.environ.get('INDEX_NAME'),
    namespace="marketing_plan",
    pinecone_api_key=os.environ.get("PINECONE_API_KEY")
  )
  print("PDF upload complete!.")

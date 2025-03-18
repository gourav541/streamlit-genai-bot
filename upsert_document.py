import os
import time
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import (
    Docx2txtLoader, PyPDFLoader, TextLoader, UnstructuredHTMLLoader
)
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone as PineconeClient, ServerlessSpec, Index

# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Function to load document based on file type
def load_document(file_path):
    _, ext = os.path.splitext(file_path.lower())
    loaders = {
        ".html": UnstructuredHTMLLoader,
        ".txt": TextLoader,
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
    }
    
    loader_class = loaders.get(ext)
    if not loader_class:
        raise ValueError(f"Unsupported document format: {ext}")
    
    print(f"📄 Loading document: {file_path}...")
    return loader_class(file_path).load()

# Function to split text into chunks
def split_text(documents, chunk_size=800, chunk_overlap=300):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    split_docs = text_splitter.split_documents(documents)
    
    # Add metadata (source file name)
    for doc in split_docs:
        doc.metadata["source"] = doc.metadata.get("source", "Unknown")
    
    print(f"✅ Total chunks created: {len(split_docs)}")
    return split_docs

# 🔹 (STEP 1) Process All Files in the Directory
directory_path = "knowledge_base/MP_cooperative_societies"
all_documents = []

for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    
    if os.path.isfile(file_path):
        try:
            content = load_document(file_path)
            all_documents.extend(content)  # Append documents from all files
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

# 🔹 (STEP 2) Split Text into Chunks
fragments = split_text(all_documents)

# 🔹 (STEP 3) Initialize OpenAI Embeddings
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("❌ Missing OpenAI API key!")

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)

# 🔹 (STEP 4) Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("❌ Missing Pinecone API key!")

pc = PineconeClient(api_key=pinecone_api_key)
index_name = "acs"

# Check if index exists & validate dimension
if index_name in pc.list_indexes().names():
    index_info = pc.describe_index(index_name)
    if index_info.dimension != 1536:
        print(f"⚠️ Index '{index_name}' has incorrect dimension ({index_info.dimension}). Recreating...")
        pc.delete_index(index_name)
        time.sleep(2)
    
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(5)
else:
    print(f"🔹 Creating new index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    time.sleep(5)

# 🔹 (STEP 5) Delete old data & Insert new embeddings
namespace = "MP_cooperative_societies_info"
index = index = pc.Index(index_name)  # Use PineconeClient to access the index

if namespace in index.describe_index_stats().get("namespaces", {}):
    index.delete(namespace=namespace, delete_all=True)
    print(f"✅ Deleted old data from namespace '{namespace}'")

# Insert new embeddings
try:
    docsearch = PineconeVectorStore.from_documents(
        documents=fragments,
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace
    )
    print("✅ Documents successfully stored in Pinecone!")
except Exception as e:
    print(f"❌ Error storing embeddings: {e}")

import os
import time
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import (
    Docx2txtLoader, PyPDFLoader, TextLoader, UnstructuredHTMLLoader
)
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone as PineconeClient, ServerlessSpec

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
    
    print(f"Loading document: {file_path}...")
    return loader_class(file_path).load()

# Function to split text into chunks
def split_text(documents, chunk_size=800, chunk_overlap=300):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    split_docs = text_splitter.split_documents(documents)
    
    # Add metadata (helps with search context)
    for doc in split_docs:
        doc.metadata["source"] = doc.metadata.get("source", "Unknown")
        doc.metadata["page"] = doc.metadata.get("page", "N/A")
    
    # Debugging: Print chunk preview
    print(f"\n✅ Total chunks created: {len(split_docs)}")
    for i, doc in enumerate(split_docs[:3]):  # Show first 3 chunks
        print(f"\n🔹 Chunk {i+1}:\n{doc.page_content[:300]}...")  # Show first 300 chars
        print(f"Metadata: {doc.metadata}")
    
    return split_docs

# 🔹 (STEP 1) Load and Process Document
file_path = "knowledge_base/MP_cooperative_societies/MP_cooperative_societies_2_chapters.pdf"
try:
    content = load_document(file_path)
    fragments = split_text(content)
except Exception as e:
    print(f"❌ Error loading document: {e}")
    exit(1)

# 🔹 (STEP 2) Initialize OpenAI Embeddings
model_name = "text-embedding-ada-002"
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("❌ Missing OpenAI API key!")

embeddings = OpenAIEmbeddings(model=model_name, openai_api_key=openai_api_key)

# 🔹 (STEP 3) Initialize Pinecone
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
        time.sleep(2)  # Allow time for deletion
    
        pc.create_index(
            name=index_name,
            dimension=1536,  
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(5)  # Wait for index creation
else:
    print(f"🔹 Creating new index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    time.sleep(5)

# 🔹 (STEP 4) Delete old data & insert new embeddings
namespace = "MP_cooperative_societies_info"
index = pc.Index(index_name)
# (STEP 4) Delete old data only if the namespace exists
if namespace in index.describe_index_stats().get("namespaces", {}):
    index.delete(namespace=namespace, delete_all=True)
    print(f"✅ Deleted old data from namespace '{namespace}'")
else:
    print(f"⚠️ Namespace '{namespace}' does not exist, skipping deletion.")

# (STEP 5) Insert new embeddings
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


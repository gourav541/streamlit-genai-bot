import os
import time
from langchain_text_splitters import MarkdownHeaderTextSplitter
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from pinecone import Pinecone , ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.vectorstores import Pinecone
from pinecone import Pinecone , ServerlessSpec

load_dotenv(find_dotenv(), override=True)


def load_document(file):
    nombre, extension = os.path.splitext(file) 
    if extension == '.html':
        from langchain.document_loaders import UnstructuredHTMLLoader
        print(f'load {file}...')
        loader = UnstructuredHTMLLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader  
        print(f'load {file}...')
        loader = TextLoader(file)
    elif extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'load {file}...')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'load {file}...')
        loader = Docx2txtLoader(file)
    else:
        print('The document format is not supported!')
        return None

    data = loader.load()
    return data

def split (data, chunk_size=512):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
    fragments = text_splitter.split_documents(data)
    return fragments


file_path = r"adagen-info-spec.pdf"
content = load_document(file_path)
fragments = split(content)

# (STEP 2) Initialize a LangChain embedding object 
model_name = "text-embedding-3-small"  
embeddings = OpenAIEmbeddings(  
    model=model_name,  
    openai_api_key=os.environ.get("OPENAI_API_KEY")  
)  

# (STEP 3) Create a serverless index  
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = "acs"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536, 
        metric="cosine", 
        spec=ServerlessSpec(
            cloud="aws", 
            region="us-east-1"
        ) 
    ) 

# (STEP 4) Embed each chunk and upsert the embeddings into a distinct namespace 
namespace = "adagen_info"

docsearch = PineconeVectorStore.from_documents(
    documents=fragments,
    index_name=index_name,
    embedding=embeddings, 
    namespace=namespace 
)

time.sleep(1)
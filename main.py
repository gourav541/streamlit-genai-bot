import os
from dotenv import load_dotenv, find_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA  
import streamlit as st
from prompts import adagen_basic_prompt

load_dotenv()

adagen_basic_prompt = adagen_basic_prompt.adagen_basics_system_prompt

st.title("Langchain Demo with OpenAI API")
query=st.text_input("Search the topic you want")

# Initialize Pinecone and get the embedding vector
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

embedding_vector = OpenAIEmbeddings().embed_query(query)

#searching the best match from VectorDB
def searching_docs():
    index = pc.Index("acs")
    answer = index.query(
        namespace="adagen_info",
        vector=embedding_vector,
        top_k=3,
        include_metadata=True,
        include_values=True
    )
    
    return answer

# Function to format the results
def format_results(data):
    formatted_output = []
    for match in data['matches']:
        metadata = match['metadata']
        text = metadata.get('text', 'No Text')
        formatted_output.append(f"\n{text}\n")
    return "\n".join(formatted_output)

# Formatting and retrieve the text
def retrieve_and_format():
    answer = searching_docs()
    formatted_text = format_results(answer)
    return formatted_text

# Refine the answer using the LLM
def refining_answer(formatted_text, query):
    llm = ChatOpenAI(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        model_name="gpt-3.5-turbo",
        temperature=0.0
    )
    
    combined_input = f"Question: {query}\n\nContext:\n{formatted_text}"
    
    messages = [
        (
            "system",
            adagen_basic_prompt,
        ),
        ("human", combined_input)
        
    ]
    
    response = llm.invoke(messages)
    content =  response.content
    return content


if __name__ == "__main__":
    formatted_text = retrieve_and_format()
    refined_response = refining_answer(formatted_text, query)
    st.success(refined_response)

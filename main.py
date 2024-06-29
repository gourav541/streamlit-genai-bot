import os
from dotenv import load_dotenv, find_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA  
import streamlit as st
from prompts import adagen_basic_prompt

load_dotenv()

adagen_basic_prompt = adagen_basic_prompt.adagen_basics_system_prompt

st.title("Dhruv - Adagen GenAI Bot")

# ---------------------------------------
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize Pinecone and get the embedding vector
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

#searching the best match from VectorDB
def searching_docs(embedding_vector):
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
def retrieve_and_format(embedding_vector):
    answer = searching_docs(embedding_vector)
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
    query = st.chat_input("Search the topic you want")
    if query:  # Check if the user has entered a query
        embedding_vector = OpenAIEmbeddings().embed_query(query)
        formatted_text = retrieve_and_format(embedding_vector)  # Make sure this function accepts embedding_vector
        response = refining_answer(formatted_text, query)

        # Display user message in chat message container
        st.chat_message("user").markdown(query)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    
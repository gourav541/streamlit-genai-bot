import os
import logging
import streamlit as st
import base64
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient
from prompts import system_prompt

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize system prompt
SYSTEM_PROMPT = system_prompt.basic_system_prompt

# Function to encode the local image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

# Get base64-encoded image
image_base64 = get_base64_image("./assets/images/ATL_PNG_FF.png")



# Create a container for the title and powered-by text
st.markdown('<div style="position: fixed;">', unsafe_allow_html=True)

# Your app title
st.title(" MP Administrative GenAI Assistant")

# Add "Powered by Adagen" text positioned at the bottom-right of the title
st.markdown(
    """
    <style>
    .powered-by {
        position: absolute;
        bottom: 0;
        right: 0;
        font-size: 15px;
        color: gray;
        font-family: Arial, sans-serif;
    }
    .logo {
        width: 50px;  /* Adjust size as needed */
        height: 50px;
        border-radius: 50%;
        margin-left: 5px;
    }
    .bottom-right-logo {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 50px;  /* Logo size at bottom-right */
        height: 50px;
        border-radius: 50%;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        z-index: 1000;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Add the powered by text inside the container
st.markdown(f"""
    <div class="powered-by">
        Powered by Adagen
        <img src="{image_base64}" class="logo">
    </div>
    """, unsafe_allow_html=True)


# Close the container
st.markdown('</div>', unsafe_allow_html=True)

# Add logo at bottom-right corner
st.markdown(
    f"""
    <img src="{image_base64}" class="bottom-right-logo">
    """,
    unsafe_allow_html=True
)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize Pinecone
try:
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_api_key:
        st.error("Pinecone API key is missing.")
        st.stop()

    pc = PineconeClient(api_key=pinecone_api_key)
except Exception as e:
    st.error(f"Error initializing Pinecone: {e}")
    logging.error(f"Pinecone initialization failed: {e}")
    st.stop()

def search_docs(embedding_vector):
    """
    Search for relevant documents in the Pinecone vector database and filter by a confidence score.
    """
    try:
        index = pc.Index("acs")
        response = index.query(
            namespace="MP_cooperative_societies_info",
            vector=embedding_vector,
            top_k=5,
            include_metadata=True,
        )

        # Prioritize metadata relevance
        return sorted(response["matches"], key=lambda x: x["score"], reverse=True)
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return None

def format_results(data):
    """
    Extracts and formats relevant text from Pinecone search results.
    """
    if not data:
        return "No relevant information found in the knowledge base."

    formatted_output = []
    for match in data:
        text_snippet = match["metadata"].get("text", "No text available").strip()
        source_link = match["metadata"].get("source", "Source not available")

        file_name = source_link.split("/")[-1] if "/" in source_link else source_link
        # Convert file path to URL if it's a local path
        if source_link.startswith("knowledge_base/MP_cooperative_societies"):
            source_link = f"{os.environ.get('SERVER_HOST')}/files/{file_name}"

        # Format response properly
        formatted_output.append(
            f"- {text_snippet}\n  **Source:** [Citation]({source_link})"
            if source_link.startswith("http")
            else f"- {text_snippet} (Source: {source_link})"
        )

    return "\n".join(formatted_output)

def get_relevant_text(embedding_vector):
    """
    Retrieve relevant documents from Pinecone and format the results.
    """
    response = search_docs(embedding_vector)
    # print('response')
    # print(response)
    return format_results(response) if response else "No relevant data available."

def generate_response(context, query):
    """
    Uses GPT-3.5 to refine the response based on retrieved context.
    """
    try:
        llm = ChatOpenAI(
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="gpt-4o",
            temperature=0.7
        )

        if "No relevant information found" in context:
            return (
                "I specialize in providing information related to M.P Cooperative Societies based on the available knowledge base. "
                "Unfortunately, I couldn't answer for your query. Kindly ask a question related to the M.P Cooperative Societies, and I will be happy to assist you."
            )
        
        combined_input = f"""
        You are an AI assistant responding to user queries based on a knowledge base.
        - **User Query:** {query}
        - **Relevant Information:**\n{context}
        - **Instructions:** Only answer based on the provided information. If it's not enough, inform the user instead of guessing.
        """

        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", combined_input)
        ]

        response = llm.invoke(messages)
        return response.content if response else "Error generating response."
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return "An error occurred while generating the response."

# Main Interaction
if __name__ == "__main__":
    query = st.chat_input("Search the topic you want")

    if query:
        logging.info(f"User Query: {query}")

        # Generate embedding vector
        embedding_vector = OpenAIEmbeddings().embed_query(query)

        # Retrieve and format relevant text
        formatted_text = get_relevant_text(embedding_vector)

        logging.info(f"Retrieved Context: {formatted_text}")

        # Generate AI response
        response = generate_response(formatted_text, query)

        # Store messages
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Display chat
        st.chat_message("user").markdown(query)
        with st.chat_message("assistant"):
            st.markdown(response, unsafe_allow_html=True)
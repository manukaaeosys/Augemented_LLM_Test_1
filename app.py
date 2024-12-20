import os
import logging
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
import streamlit as st
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. User Query Input
def handle_user_query(user_query):
    """Handles user query by interacting with the retrieval and LLM components."""
    if not user_query:
        st.warning("Please enter a query.")
        return

    try:
        retriever_type = st.radio("Choose Retrieval Method", ["Keyword Search", "Embedding-Based"])
        if retriever_type == "Keyword Search":
            relevant_chunks = keyword_search_retriever(st.session_state.text_chunks, user_query)
            augmented_prompt = f"User Query: {user_query}\nRelevant Context:\n" + "\n".join(relevant_chunks)
        else:
            augmented_prompt = augment_user_prompt(user_query, st.session_state.retriever)

        response = generate_response(augmented_prompt)
        st.session_state.chat_history.append((user_query, response))

        for user_msg, bot_msg in st.session_state.chat_history:
            st.write(f"**User:** {user_msg}")
            st.write(f"**Bot:** {bot_msg}")
    except Exception as e:
        logger.error(f"Error processing user query: {e}")
        st.error("An error occurred while processing your query.")

# 2.1. Keyword Search Retrieval
def keyword_search_retriever(text_chunks, user_query):
    """Retrieves text chunks matching keywords in the user query."""
    try:
        keywords = user_query.lower().split()  # Split query into keywords
        relevant_chunks = [
            chunk for chunk in text_chunks
            if any(keyword in chunk.lower() for keyword in keywords)
        ]
        return relevant_chunks
    except Exception as e:
        logger.error(f"Error in keyword search retrieval: {e}")
        return []

# 2.2. Embedding-Based Retrieval
def retrieve_data(vectorstore):
    """Creates a retriever for querying relevant documents."""
    try:
        return vectorstore.as_retriever()
    except Exception as e:
        logger.error(f"Error initializing retriever: {e}")
        return None

# 3. Document Chunking and Embedding
def process_documents(pdf_files):
    """Processes uploaded PDF documents and returns text chunks."""
    text = ""
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_pdf, pdf_files)
        text = "".join(results)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=100
    )
    text_chunks = text_splitter.split_text(text)
    st.session_state.text_chunks = text_chunks  # Save for keyword search

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    try:
        vectorstore = OpenAI.from_texts(texts=text_chunks, embedding=embeddings)
        st.session_state.vectorstore = vectorstore  # Save for embedding-based retrieval
        return vectorstore
    except Exception as e:
        logger.error(f"Error creating vectorstore: {e}")
        return None

def process_pdf(pdf):
    """Extracts text from a single PDF."""
    pdf_text = ""
    try:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()
    except Exception as e:
        logger.error(f"Error reading PDF {pdf.name}: {e}")
    return pdf_text

# 4. Augmentation of User Prompt
def augment_user_prompt(user_query, retriever):
    """Augments user prompt with relevant contextual information from retrieved documents."""
    try:
        relevant_documents = retriever.retrieve(user_query)
        augmented_prompt = f"User Query: {user_query}\nRelevant Context:\n" + "\n".join([doc.page_content for doc in relevant_documents])
        return augmented_prompt
    except Exception as e:
        logger.error(f"Error augmenting user prompt: {e}")
        return user_query

# 5. Response Generation
def generate_response(augmented_prompt):
    """Generates a response using the LLM based on the augmented prompt."""
    try:
        llm = ChatOpenAI(model="gpt-4")
        return llm.generate([augmented_prompt])[0].text
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "An error occurred while generating a response."

# Main Application Logic
def main():
    load_dotenv()
    st.title("Chat with Your Documents")

    # Session state initialization
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User query input
    user_query = st.text_input("Ask a question about your documents:")
    if user_query:
        handle_user_query(user_query)

    # Sidebar: Document upload and processing
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_files = st.file_uploader("Upload your PDFs", accept_multiple_files=True)

        if st.button("Process"):
            if not pdf_files:
                st.error("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing your documents..."):
                    vectorstore = process_documents(pdf_files)

                    if vectorstore:
                        retriever = retrieve_data(vectorstore)
                        st.session_state.retriever = retriever  # Save for embedding-based retrieval
                        st.success("Documents processed successfully! You can now ask questions.")
                    else:
                        st.error("Failed to process documents.")

if __name__ == '__main__':
    main()

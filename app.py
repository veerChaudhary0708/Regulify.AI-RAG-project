# importing all the nacessory libraries and frameworks
import os
import tempfile

import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize Streamlit page config at the top level
st.set_page_config(page_title="Math-Aware PDF RAG", page_icon="🧮", layout="wide")

@st.cache_resource(show_spinner=False)
def process_document(uploaded_file, api_key: str):
    """
    Handles the ingestion, chunking, and embedding of the uploaded PDF.
    Cached by Streamlit to prevent re-running the embedding model on every UI interaction.
    """
    # PyPDFLoader requires a physical file path, so we write the uploaded byte stream to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # We use a large chunk overlap here to ensure that multiline LaTeX equations, 
        # matrices, and theorems are not split across different chunks.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200, 
            chunk_overlap=250,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        # I am using google embedding model(because I have googel AI studio subscription),
        # we can use OpenAI's model or even opensource models from huggingface
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=api_key, 
            model="models/gemini-embedding-2" 
        )
        
        # FAISS is chosen for its lightweight, in-memory footprint, making this app easy to run locally
        # If we have large data and require frequent retrivels, we can also use a dadicated vector database.
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store
        
    finally:
        #Ensure cleanup of the temporary file to prevent memory/storage leaks(got to know after a issue in earlier projects)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def main():
    st.title("🧮 Regulify.AI - Math-Aware RAG Assistant")
    st.markdown("Upload a complex mathematical or technical PDF and ask questions about it.")

    # Sidebar: Credentials and Architecture details
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("Google Gemini API Key", type="password", help="Enter your Google AI Studio API Key.")
        st.markdown("---")
        st.markdown("**Architecture:**\n- UI: Streamlit\n- Vector Store: FAISS\n- Embeddings: `gemini-embedding-2`\n- LLM: `gemini-3.5-flash`\n- Strategy: Recursive chunking optimized for mathematical context.")

    # Initialize session state variables for chat history and the vector store
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not api_key:
        st.warning("👈 Please enter your Google Gemini API Key in the sidebar to begin.")
        return # stop execution until key is provided

    uploaded_file = st.file_uploader("Upload your PDF document (like lintransf.pdf)", type="pdf")

    if uploaded_file is not None:
        # Process the document only if it hasn't been processed yet in this session
        if st.session_state.vector_store is None:
            with st.spinner("Processing document and generating vector embeddings..."):
                st.session_state.vector_store = process_document(uploaded_file, api_key)
            st.success("Document processed successfully! You can now chat with it.")

        #Render existing chathistory
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat interface logic
        if user_input := st.chat_input("Ask a question about linear transformations, rank-nullity, etc..."):
            
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # System prompt engineered specifically to preserve mathematical integrity
                    system_prompt = (
                        "You are an expert mathematical and engineering AI assistant. "
                        "Use the provided context to answer the user's question accurately. "
                        "The context contains complex linear algebra, theorems, and LaTeX-style math. "
                        "CRITICAL: Preserve all mathematical notation. Format mathematical formulas clearly using Markdown/LaTeX. "
                        "When explaining theorems or matrix transformations, always state the broader conceptual conclusion or purpose of the math. "
                        "If the answer is not contained in the context, politely state that you do not know based on the provided text.\n\n"
                        "Context:\n{context}"
                    )
                    
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ])

                    llm = ChatGoogleGenerativeAI(
                        google_api_key=api_key,
                        model="gemini-3.5-flash",
                        temperature=0.1, # Low temperature to minimize hallucinations on math facts
                        convert_system_message_to_human=True
                    )

                    # Retrieval and chain assembly
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 8})
                    question_answer_chain = create_stuff_documents_chain(llm, prompt)
                    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                    response = rag_chain.invoke({"input": user_input})
                    answer = response["answer"]
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
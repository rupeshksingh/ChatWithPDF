import streamlit as st
from langchain_community.llms import HuggingFaceEndpoint
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain.callbacks import StdOutCallbackHandler
from api_key import api_key

# Streamlit interface
st.title("PDF Chat with RAG Model")
st.write("Upload a PDF and ask questions about its content.")

# File upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Process the PDF if uploaded
if uploaded_file:
    with open("uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load and process the PDF
    loader = PyPDFLoader("uploaded_file.pdf")
    documents = loader.load_and_split()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    retriever = vectorstore.as_retriever()
    
    # Store the retriever in the session state
    st.session_state["retriever"] = retriever
    st.success("PDF processed successfully. You can now ask questions.")

# Query input
query = st.text_input("Enter your query here")

if query and "retriever" in st.session_state:
    retriever = st.session_state["retriever"]
    
    # Load the LLM
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, max_new_tokens=1024, temperature=1, huggingfacehub_api_token=api_key
    )
    
    # Create the QA chain
    qa_chain_with_sources = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        callbacks=[StdOutCallbackHandler()],
        return_source_documents=True
    )
    
    # Get the response
    response = qa_chain_with_sources({"query": query})
    
    # Display the response
    st.write("## Response")
    st.write(response["result"])
    
    # Optionally display the source documents
    st.write("## Source Documents")
    for source_doc in response["source_documents"]:
        st.write(source_doc.page_content)
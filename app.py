import streamlit as st
from langchain_community.llms import HuggingFaceEndpoint
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain.callbacks import StdOutCallbackHandler

# Streamlit interface
st.title("PDF Chat with RAG Model")
st.write("Upload a PDF and ask questions about its content.")

# File upload
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])

# File upload
if uploaded_file:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    with open(f"uploaded_file.{file_extension}", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if file_extension == "pdf":
        loader = PyPDFLoader(f"uploaded_file.{file_extension}")
    elif file_extension == "docx":
        loader = Docx2txtLoader(f"uploaded_file.{file_extension}")
    elif file_extension == "txt":
        loader = TextLoader(f"uploaded_file.{file_extension}")
    else:
        st.error("Unsupported file type")

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
        repo_id=repo_id, max_new_tokens=1024, temperature=1, huggingfacehub_api_token=st.secrets['api_key']
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

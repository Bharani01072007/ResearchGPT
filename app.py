import streamlit as st
import os
import tempfile
import re
import fitz
from PIL import Image
import shutil
from dotenv import load_dotenv

@st.cache_data
def get_pdf_page_image(pdf_path, page_num):
    try:
        # Resolve the fallback path if only the basename is recorded
        if not os.path.exists(pdf_path):
            basename = os.path.basename(pdf_path)
            fallback_path = os.path.join(os.getcwd(), "pdf_uploads", basename)
            if os.path.exists(fallback_path):
                pdf_path = fallback_path
            else:
                return None
                
        doc = fitz.open(pdf_path)
        # PyMuPDF expects 0-indexed pages but Streamlit provides 1-indexed
        page_index = max(0, int(page_num) - 1)
        page = doc.load_page(page_index)
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img
    except Exception as e:
        print(f"Error rendering PDF: {e}")
        return None

# Import our custom modules
from loader import load_documents_from_paths
from chunking import split_documents
from embeddings import get_huggingface_embeddings, get_gemini_embeddings
from vectorstore import create_and_save_vectorstore, load_vectorstore, clear_vectorstore
from retriever import get_retriever
from rag_pipeline import generate_answer

# Load environment variables
load_dotenv()

st.set_page_config(page_title="ResearchGPT", page_icon="📄", layout="wide")

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "total_pages" not in st.session_state:
    st.session_state.total_pages = 0

# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ Configurations")
    
    api_key_input = os.getenv("GEMINI_API_KEY", "")
    model_name_choice = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
    
    st.markdown("---")
    st.subheader("Document Processing")
    
    embedding_strategy = st.radio(
        "Embedding Model",
        ["HuggingFace (BAAI/bge-small-en) - Open Source", "Google Gemini (gemini-embedding-001) - Commercial"],
        index=1,
        help="Select the embedding model. Re-process documents if you change this."
    )
    
    chunking_strategy = st.radio(
        "Chunking Strategy",
        ["A (1000 size / 200 overlap)", "B (1500 size / 300 overlap)"],
        index=0,
        help="Strategy to split document text. Strategy B keeps more context together for code elements."
    )
    strategy_map = {"A (1000 size / 200 overlap)": "A", "B (1500 size / 300 overlap)": "B"}
    
    # File uploader moved to main UI

    st.markdown("---")
    st.subheader("Retrieval Settings")
    retrieval_strategy = st.radio(
        "Retrieval Strategy",
        [
            "Similarity", 
            "MMR (Max Marginal Relevance)",
            "Hybrid Search (BM25 + Semantic)",
            "Reranker (MMR + Cross-Encoder)"
        ],
        index=0,
        help="How to fetch chunks from FAISS."
    )
    retrieval_map = {
        "Similarity": "similarity", 
        "MMR (Max Marginal Relevance)": "mmr",
        "Hybrid Search (BM25 + Semantic)": "hybrid",
        "Reranker (MMR + Cross-Encoder)": "reranker"
    }
    
    st.markdown("---")
    if st.session_state.total_pages > 0:
        st.info(f"Total Pages in uploaded PDFs: {st.session_state.total_pages}")

# Main UI
st.title("📄 ResearchGPT - PDF QA Agent")
st.markdown("Ask anything about your uploaded documents. ResearchGPT will strictly use the context to answer and provide citations.")

st.subheader("📁 Upload Documents")
enable_multimodal = st.checkbox("Enable Image Extraction (Slower)", value=False, help="Uses Vision AI to extract text from images in the PDF. Disable this to speed up document processing.")

uploaded_files = st.file_uploader(
    "Upload Research Papers (PDF)", 
    type="pdf", 
    accept_multiple_files=True
)

col1, col2 = st.columns(2)
with col1:
    if st.button("Process Documents", use_container_width=True):
        if not uploaded_files:
            st.error("Please upload at least one PDF.")
        else:
            with st.spinner("Processing Documents..."):
                upload_dir = os.path.join(os.getcwd(), "pdf_uploads")
                os.makedirs(upload_dir, exist_ok=True)
                file_paths = []
                total_pages = 0
                for idx, file in enumerate(uploaded_files):
                    temp_path = os.path.join(upload_dir, file.name)
                    with open(temp_path, "wb") as f:
                        f.write(file.getvalue())
                    file_paths.append(temp_path)
                    
                    try:
                        with fitz.open(temp_path) as doc:
                            total_pages += len(doc)
                    except Exception as e:
                        print(f"Error reading page count for {file.name}: {e}")
                
                st.session_state.total_pages = total_pages
                
                # 1. Load
                if enable_multimodal:
                    st.toast("Loading PDFs with Multimodal extraction (This may take a while)...")
                else:
                    st.toast("Loading PDFs...")
                docs = load_documents_from_paths(file_paths, api_key=api_key_input, extract_images=enable_multimodal)
                
                # 2. Chunk
                st.toast("Chunking Text...")
                chunks = split_documents(docs, strategy=strategy_map[chunking_strategy])
                
                # 3. Embed & 4. VectorStore
                st.toast("Generating Embeddings & FAISS Index...")
                if "HuggingFace" in embedding_strategy:
                    embeddings = get_huggingface_embeddings()
                else:
                    embeddings = get_gemini_embeddings(api_key=api_key_input)
                    
                vectorstore = create_and_save_vectorstore(chunks, embeddings)
                st.session_state.vectorstore = vectorstore
                st.success("Documents Embedded Successfully!")

with col2:
    if st.button("Clear Documents/Reset", use_container_width=True):
        clear_vectorstore()
        upload_dir = os.path.join(os.getcwd(), "pdf_uploads")
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
        st.session_state.vectorstore = None
        st.session_state.chat_history = []
        st.session_state.total_pages = 0
        st.toast("Documents and chat history cleared!")
        st.rerun()

st.divider()

chat_container = st.container(height=500)

# Display Chat History
with chat_container:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
            if "citations" in message and message["citations"]:
                st.markdown("##### 📚 Reference Documents")
                for idx, doc in enumerate(message["citations"]):
                    source = doc.metadata.get('source', 'Unknown')
                    filename = os.path.basename(source) if source != 'Unknown' else 'Unknown'
                    page = doc.metadata.get('page', 'Unknown')
                    content = re.sub(r'\n{3,}', '\n\n', doc.page_content).strip()
                    with st.expander(f"📄 [{idx+1}] {filename} (Page {page})"):
                        st.markdown(f"{content}")
                        
                        if st.toggle("📸 View Original Page", key=f"history_toggle_{len(st.session_state.chat_history)}_{idx}_{page}_{filename}"):
                            img = get_pdf_page_image(source, page)
                            if img:
                                st.image(img, caption=f"{filename} - Page {page}", use_container_width=True)
                            else:
                                st.error("Document is no longer available on disk. Please re-upload.")

# Input from user
prompt = st.chat_input("Ask a question about the papers...")

if prompt:
    # Append user question to history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)
    
        with st.chat_message("assistant"):
            if not api_key_input:
                st.warning("Please enter your Gemini API Key in the sidebar.")
            elif st.session_state.vectorstore is None:
                # Try to load existing local FAISS vectorstore
                with st.spinner("Checking local FAISS index..."):
                    if "HuggingFace" in embedding_strategy:
                        embeddings = get_huggingface_embeddings()
                    else:
                        embeddings = get_gemini_embeddings(api_key=api_key_input)
                    vs = load_vectorstore(embeddings)
                    if vs:
                        st.session_state.vectorstore = vs
                    
                if st.session_state.vectorstore is None:
                    st.warning("Please upload and process documents first.")
            
            if api_key_input and st.session_state.vectorstore is not None:
                with st.spinner("Generating Response..."):
                    # Configure Retriever
                    retriever = get_retriever(
                        st.session_state.vectorstore, 
                        strategy=retrieval_map[retrieval_strategy],
                        top_k=3 # Enforce top 3 according to requirements
                    )
                    
                    try:
                        # Generate Answer
                        answer, retrieved_docs = generate_answer(
                            query=prompt, 
                            retriever=retriever, 
                            api_key=api_key_input,
                            model_name=model_name_choice,
                            chat_history=st.session_state.chat_history
                        )
                        
                        # Highlight 'Note:' segments in a red container with an asterisk
                        formatted_answer = re.sub(
                            r"(?im)^(?:\*\*?)?(?:Note|NOTE):?(?:\*\*?)?\s*(.+)$",
                            r'<div style="background-color: #ffecec; border-left: 5px solid #ff4b4b; color: #b71c1c; padding: 10px; border-radius: 5px; margin: 10px 0;"><strong style="font-size: 1.2em;">*</strong> <strong>Note:</strong> \1</div>',
                            answer
                        )
                        
                        st.markdown(formatted_answer, unsafe_allow_html=True)
                        
                        # If model doesn't know, clear retrieved docs so we don't cite them
                        if "I don't know" in formatted_answer:
                            retrieved_docs = []
                        
                        # Display citations inline cleanly
                        if retrieved_docs:
                            st.markdown("### 📚 Citing Sources")
                            for idx, doc in enumerate(retrieved_docs):
                                source = doc.metadata.get('source', 'Unknown')
                                filename = os.path.basename(source) if source != 'Unknown' else 'Unknown'
                                page = doc.metadata.get('page', 'Unknown')
                                content = re.sub(r'\n{3,}', '\n\n', doc.page_content).strip()
                                
                                with st.expander(f"📄 Source {idx+1}: {filename} (Page {page})"):
                                    st.markdown(content)
                                    
                                    if st.toggle("📸 View Original Page", key=f"live_toggle_{idx}_{page}_{filename}"):
                                        img = get_pdf_page_image(source, page)
                                        if img:
                                            st.image(img, caption=f"{filename} - Page {page}", use_container_width=True)
                                        else:
                                            st.error("Document is no longer available on disk. Please re-upload.")
                            
                        # Save to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": formatted_answer,
                            "citations": retrieved_docs
                        })
                    except Exception as e:
                        st.error(f"Error communicating with Gemini API: {e}")

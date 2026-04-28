# ResearchGPT

ResearchGPT is a local Retrieval-Augmented Generation (RAG) agent that allows you to chat with your PDF research papers. It is built using Streamlit, LangChain, FAISS, and Google's Gemini models. The system strictly answers questions based on the provided document context and features verifiable inline citations with live page rendering.

## Setup Instructions

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment:**
   Create a `.env` file in the root directory and add your Gemini API Key (or input it directly in the app sidebar):
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

3. **Run the Application:**
   ```bash
   python -m streamlit run app.py
   ```

---

## 📊 Method Comparisons

The application offers multiple configurable methods and strategies directly in the sidebar. Below is a comparison of these methods to help you choose the best configuration for your documents.

### 1. Document Extraction Methods

| Method | Speed | Best For | Description |
| :--- | :--- | :--- | :--- |
| **Text-Only Extraction (Default)** | ⚡ Very Fast | Standard academic papers, heavy text, and large tables. | Uses `pymupdf4llm` to instantly convert PDF pages into structured Markdown text. Tables are preserved perfectly. |
| **Multimodal Image Extraction** | 🐢 Slow | Papers where charts, graphs, or visual diagrams are critical. | Extracts every image/graph from the PDF and uses Gemini Vision AI to generate a detailed text summary of the image data, which is then added to the vector database. |

### 2. Embedding Models

| Model | Cost / Privacy | Strengths | Description |
| :--- | :--- | :--- | :--- |
| **HuggingFace (BAAI/bge-small-en)** | Free / 100% Local | Zero API costs, complete privacy, very fast on CPU. | An open-source, highly efficient embedding model. It runs locally and provides excellent semantic mapping for general English text. |
| **Google Gemini (gemini-embedding-001)** | API Costs | Deep semantic understanding, complex reasoning. | A commercial embedding model. While it provides slightly better context mapping for highly complex or nuanced queries, it requires sending data to Google's API and is subject to rate limits. |

### 3. Chunking Strategies

When documents are loaded, they must be split into smaller "chunks" so the LLM can process them efficiently.

| Strategy | Size / Overlap | Best For | Description |
| :--- | :--- | :--- | :--- |
| **Strategy A** | 1000 / 200 | General Q&A, standard text paragraphs. | Creates smaller, highly specific chunks. This ensures that the vector database returns very concentrated snippets of information. |
| **Strategy B** | 1500 / 300 | Code snippets, long continuous arguments. | Creates larger chunks. This is useful if your research papers contain long blocks of code or complex mathematical proofs that shouldn't be split in half. |

### 4. Retrieval Strategies

When you ask a question, the system searches the FAISS vector database to find the most relevant chunks.

| Strategy | Goal | How It Works |
| :--- | :--- | :--- |
| **Similarity** | Highest Relevance | Simply returns the top `K` chunks that have the closest mathematical vector distance to your question. Can sometimes return highly redundant chunks if the same concept is repeated in the text. |
| **MMR (Max Marginal Relevance)** | Relevance + Diversity | Fetches a larger pool of similar chunks, but then penalizes chunks that are too similar to each other. Ensures the LLM gets diverse perspectives from the document to form its answer. |

---

## Project Structure
- `app.py`: Main Streamlit frontend interface.
- `rag_pipeline.py`: LangChain logic for generating answers from context.
- `loader.py`: Handles PDF parsing and optional Gemini Vision image summarization.
- `chunking.py`: Splits documents into manageable sizes.
- `embeddings.py`: Handles local HuggingFace and remote Gemini embedding models.
- `vectorstore.py`: Manages the local FAISS database with batching and rate-limit handling.
- `retriever.py`: Configures standard Similarity and MMR retrieval.

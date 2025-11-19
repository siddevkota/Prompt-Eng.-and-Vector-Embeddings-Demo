# 🔍 Vector Embeddings & RAG Demo

Interactive Streamlit application demonstrating vector embeddings, semantic search, and Retrieval-Augmented Generation (RAG) using OpenAI embeddings and FAISS.

## 🎯 Features

### 1. **Generate Embeddings**
- Convert text to high-dimensional vectors (1536 dimensions)
- Use OpenAI's `text-embedding-3-small` model
- Process multiple texts in batch
- View embedding statistics and metrics

### 2. **Vector Search**
- Build FAISS index for efficient similarity search
- Find semantically similar texts using cosine similarity
- Adjust number of results (top-k)
- See similarity scores and rankings

### 3. **Visualize Embeddings**
- **2D/3D Visualization**: PCA and t-SNE dimensionality reduction
- **Interactive Plots**: Explore embeddings in lower dimensions
- **Heatmap**: Compare similarity between all text pairs
- **Clustering**: Discover natural groupings in your data

### 4. **RAG Pipeline**
- Upload or input documents
- Automatic text chunking with overlap
- Build vector store from documents
- Ask questions and get AI-generated answers with source references
- Compare different chunking strategies

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key

### Setup

1. **Install dependencies** (from main project folder):
   ```bash
   pip install -r ../requirements.txt
   ```

2. **Configure API key:**
   ```bash
   cp .env.example .env
   # Edit .env and add: OPENAI_API_KEY=sk-your-key-here
   ```

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

4. **Open browser:** Navigate to `http://localhost:8501`

## 📚 How to Use

### Tab 1: Generate Embeddings
1. Enter your texts (one per line) or use the default examples
2. Click "Generate Embeddings"
3. View embedding dimensions and statistics
4. Embeddings are stored in session for use in other tabs

### Tab 2: Vector Search
1. First generate embeddings in Tab 1
2. Click "Build FAISS Index" to create searchable index
3. Enter a search query
4. Adjust top-k slider for number of results
5. View ranked results with similarity scores

### Tab 3: Visualize Embeddings
1. Choose visualization method (PCA or t-SNE)
2. Select 2D or 3D projection
3. Explore interactive plots with hover information
4. View similarity heatmap to compare all texts
5. Identify clusters and relationships

### Tab 4: RAG Pipeline
1. Input your document text or use sample data
2. Configure chunk size and overlap
3. Click "Process Document" to create vector store
4. Ask questions about the document
5. Get AI-generated answers with retrieved context
6. See which chunks were used to answer

## 🎓 Key Concepts

### Vector Embeddings
Text is converted into numerical vectors that capture semantic meaning. Similar texts have similar vectors.

### Semantic Search
Unlike keyword search, semantic search finds results based on meaning, not just word matches.

### FAISS (Facebook AI Similarity Search)
High-performance library for efficient similarity search in large vector collections.

### RAG (Retrieval-Augmented Generation)
Combines vector search with LLMs:
1. Split documents into chunks
2. Convert chunks to embeddings
3. Find relevant chunks for a query
4. Use chunks as context for LLM to generate answers

## 🔧 Technical Details

- **Embedding Model**: OpenAI `text-embedding-3-small` (1536 dimensions)
- **Vector Store**: FAISS with L2 distance
- **Chunking**: LangChain's RecursiveCharacterTextSplitter
- **Visualization**: Plotly for interactive charts
- **Dimensionality Reduction**: PCA (fast) and t-SNE (detailed)

## 📊 Included Guides

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: System architecture and data flow diagrams
- **[COMPARISON_GUIDE.md](COMPARISON_GUIDE.md)**: Detailed comparisons of embedding techniques and search methods

## 💡 Tips

1. **Start small**: Test with 5-10 texts first
2. **PCA vs t-SNE**: Use PCA for quick overview, t-SNE for detailed clustering
3. **Chunk size**: Experiment with 200-500 characters for RAG
4. **Overlap**: Use 20-30% overlap to maintain context between chunks
5. **Similarity scores**: Scores > 0.8 indicate very similar content

## 🛠️ Troubleshooting

**"Please set OPENAI_API_KEY"**
- Check your `.env` file has the correct API key

**"Generate embeddings first"**
- Visit Tab 1 and create embeddings before using other features

**Visualization is slow**
- Reduce number of texts or use PCA instead of t-SNE

**RAG answers are generic**
- Try smaller chunk sizes or better document formatting

## 📖 Learn More

- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [Understanding Vector Embeddings](https://www.pinecone.io/learn/vector-embeddings/)

## 🎨 Sample Use Cases

- **Document Q&A**: Upload company docs and ask questions
- **Semantic Search**: Find similar customer reviews or feedback
- **Content Clustering**: Group similar articles or emails
- **Recommendation**: Find similar products or content
- **Duplicate Detection**: Identify near-duplicate texts

---

**Happy Exploring! 🚀**

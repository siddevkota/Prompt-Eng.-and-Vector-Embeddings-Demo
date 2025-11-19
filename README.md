# 🤖 Prompt Engineering & Vector Embeddings Demo

Interactive demos exploring LLM fundamentals, prompt engineering techniques, and vector embeddings with practical examples using OpenAI API.

## 📂 Project Structure

```
├── prompt_engineering/     # Week 1: Prompt Engineering & LLM Basics
│   └── app.py             # Streamlit app with prompt techniques
├── embeddings/            # Week 2: Vector Embeddings & RAG
│   ├── app.py            # Embeddings visualization & search
│   └── utils.py          # Helper functions
├── requirements.txt       # Python dependencies
└── pyproject.toml        # Project configuration
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/siddevkota/Prompt-Eng.-and-Vector-Embeddings-Demo.git
   cd Prompt-Eng.-and-Vector-Embeddings-Demo
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API key:**
   ```bash
   # Create .env file in each module
   cp prompt_engineering/.env.example prompt_engineering/.env
   cp embeddings/.env.example embeddings/.env
   
   # Edit .env files and add your OpenAI API key
   ```

4. **Run the demos:**
   ```bash
   # Prompt Engineering Demo
   streamlit run prompt_engineering/app.py
   
   # Vector Embeddings Demo
   streamlit run embeddings/app.py
   ```

## 🎯 Features

### Prompt Engineering Module
- ✨ Zero-shot, few-shot, and role-based prompting
- 🎛️ LLM parameter tuning (temperature, top-p, max tokens)
- 🔧 Function/tool calling demonstrations
- 📊 Side-by-side response comparison

### Vector Embeddings Module
- 🔍 Text-to-vector embedding generation
- 📐 Similarity search with FAISS
- 📊 2D/3D visualization (PCA, t-SNE)
- 🤖 RAG pipeline with document Q&A
- 🆚 Embedding model comparison

## 📚 Learning Outcomes

- Understand effective prompt design patterns
- Master LLM parameter tuning for different use cases
- Learn vector embeddings and semantic search
- Build RAG applications for document Q&A
- Compare different embedding models and techniques

## 🛠️ Tech Stack

- **Framework:** Streamlit
- **LLM API:** OpenAI (GPT-3.5/4, text-embedding-3)
- **Vector Search:** FAISS
- **ML Tools:** LangChain, scikit-learn, NumPy, Pandas
- **Visualization:** Plotly, Matplotlib

## 📖 Documentation

- [Prompt Engineering README](prompt_engineering/README.md)
- [Embeddings Architecture](embeddings/ARCHITECTURE.md)
- [Embedding Comparison Guide](embeddings/COMPARISON_GUIDE.md)

## 🔒 Security Note

Never commit your `.env` files or API keys. Use the provided `.env.example` templates.

## 📝 License

Educational demo project for learning purposes.

---

**Happy Learning! 🚀**
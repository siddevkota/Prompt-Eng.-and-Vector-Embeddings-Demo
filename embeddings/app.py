import streamlit as st
import openai
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Tuple
import faiss
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Week 2: Embeddings & Vector Search",
    page_icon="🔍",
    layout="wide"
)

api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = api_key

if 'embeddings_data' not in st.session_state:
    st.session_state.embeddings_data = None
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'documents' not in st.session_state:
    st.session_state.documents = []

st.title("🔍 Week 2: Embeddings & Vector Databases")

tab1, tab2, tab3, tab4 = st.tabs([
    "Generate Embeddings",
    "Vector Search",
    "Visualize Embeddings",
    "RAG Pipeline"
])

with tab1:
    st.header("Generate Text Embeddings")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        default_texts = """Machine learning is a subset of artificial intelligence.
Deep learning uses neural networks with multiple layers.
Natural language processing helps computers understand human language.
Computer vision enables machines to interpret visual information.
Reinforcement learning trains agents through rewards and penalties.
Transfer learning applies knowledge from one task to another.
Supervised learning uses labeled data for training.
Unsupervised learning finds patterns in unlabeled data.
Neural networks are inspired by biological neurons.
Transformers revolutionized natural language processing."""
        
        text_input = st.text_area(
            "Enter text (one sentence per line)",
            value=default_texts,
            height=300
        )
        
        if st.button("Generate Embeddings", type="primary"):
            if not api_key:
                st.error("Please set OPENAI_API_KEY in .env file!")
            else:
                with st.spinner("Generating embeddings..."):
                    try:
                        texts = [line.strip() for line in text_input.split('\n') if line.strip()]
                        
                        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
                        embeddings = []
                        
                        for text in texts:
                            embedding = embeddings_model.embed_query(text)
                            embeddings.append(embedding)
                        
                        st.session_state.embeddings_data = {
                            'texts': texts,
                            'embeddings': np.array(embeddings)
                        }
                        
                        st.success(f"✅ Generated {len(texts)} embeddings!")
                        st.info(f"Embedding dimension: {len(embeddings[0])}")
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with col2:
        if st.session_state.embeddings_data:
            data = st.session_state.embeddings_data
            st.metric("Number of texts", len(data['texts']))
            st.metric("Embedding dimension", data['embeddings'].shape[1])
            st.metric("Total vectors", data['embeddings'].shape[0])
        else:
            st.info("Generate embeddings to see information")

with tab2:
    st.header("Semantic Search with FAISS")
    
    if not st.session_state.embeddings_data:
        st.warning("⚠️ Please generate embeddings first in the 'Generate Embeddings' tab!")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("Build FAISS Index"):
                with st.spinner("Building FAISS index..."):
                    try:
                        data = st.session_state.embeddings_data
                        texts = data['texts']
                        embeddings = data['embeddings']
                        
                        documents = [
                            Document(page_content=text, metadata={"index": i})
                            for i, text in enumerate(texts)
                        ]
                        
                        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
                        vectorstore = LangChainFAISS.from_documents(
                            documents,
                            embeddings_model
                        )
                        
                        st.session_state.vectorstore = vectorstore
                        st.session_state.documents = documents
                        
                        st.success("✅ FAISS index built successfully!")
                        st.info(f"Indexed {len(documents)} documents")
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        with col2:
            if st.session_state.vectorstore:
                st.success("✅ Index is ready")
                st.metric("Documents indexed", len(st.session_state.documents))
            else:
                st.info("Build the index first")
        
        st.markdown("---")
        
        if st.session_state.vectorstore:
            query = st.text_input(
                "Enter your search query",
                placeholder="e.g., 'How do neural networks work?'"
            )
            
            k = st.slider("Number of results", min_value=1, max_value=10, value=3)
            
            if query and st.button("Search", type="primary"):
                with st.spinner("Searching..."):
                    try:
                        results = st.session_state.vectorstore.similarity_search_with_score(
                            query, k=k
                        )
                        
                        st.subheader(f"Top {k} Results")
                        
                        for i, (doc, score) in enumerate(results, 1):
                            similarity = 1 / (1 + score)
                            
                            with st.expander(f"Result {i} - Similarity: {similarity:.4f}"):
                                st.write(f"**Text:** {doc.page_content}")
                                st.write(f"**Score:** {similarity:.4f}")
                                st.progress(similarity)
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

with tab3:
    st.header("Embedding Space Visualization")
    
    if st.checkbox("Show test chart (debug)", value=True):
        try:
            import plotly.express as px
            st.write("Plotly imported successfully!")
            test_df = pd.DataFrame({
                'x': [1, 2, 3, 4, 5],
                'y': [1, 4, 9, 16, 25]
            })
            test_fig = px.scatter(test_df, x='x', y='y', title="Test Chart")
            test_fig.update_layout(width=600, height=400)
            st.plotly_chart(test_fig, use_container_width=False)
            st.success("If you see a chart above, Plotly is working!")
        except Exception as e:
            st.error(f"Plotly test failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    if not st.session_state.embeddings_data:
        st.warning("⚠️ Please generate embeddings first!")
    else:
        data = st.session_state.embeddings_data
        texts = data['texts']
        embeddings = data['embeddings']
        
        # Show heatmap immediately when embeddings exist
        st.subheader("Cosine Similarity Heatmap")
        
        try:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized = embeddings / norms
            similarity_matrix = np.dot(normalized, normalized.T)
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=similarity_matrix,
                x=[f"T{i+1}" for i in range(len(texts))],
                y=[f"T{i+1}" for i in range(len(texts))],
                colorscale='RdBu',
                zmid=0
            ))
            
            fig_heatmap.update_layout(
                title="Cosine Similarity Matrix",
                height=500,
                width=700
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True, key="heatmap")
        except Exception as e:
            st.error(f"Heatmap error: {str(e)}")
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            method = st.selectbox("Dimensionality Reduction", ["PCA", "t-SNE"], key="method_select")
            dims = st.radio("Dimensions", ["2D", "3D"], key="dims_select")
            
            if st.button("Generate Visualization", type="primary", key="viz_button"):
                with st.spinner(f"Computing {method}..."):
                    try:
                        n_components = 3 if dims == "3D" else 2
                        
                        if method == "PCA":
                            reducer = PCA(n_components=n_components)
                            reduced = reducer.fit_transform(embeddings)
                            variance = reducer.explained_variance_ratio_
                            st.info(f"Explained variance: {variance.sum():.2%}")
                        else:
                            perplexity = min(30, len(texts) - 1)
                            reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
                            reduced = reducer.fit_transform(embeddings)
                        
                        st.session_state.reduced_embeddings = reduced
                        st.session_state.reduction_method = method
                        st.session_state.dims = dims
                        st.success("✅ Done!")
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # Show reduced embeddings plot
        with col2:
            if 'reduced_embeddings' in st.session_state:
                try:
                    reduced = st.session_state.reduced_embeddings
                    method = st.session_state.reduction_method
                    dims = st.session_state.dims
                    
                    current_texts = st.session_state.embeddings_data['texts']
                    
                    if dims == "2D":
                        df = pd.DataFrame({
                            'x': reduced[:, 0],
                            'y': reduced[:, 1],
                            'text': current_texts
                        })
                        
                        fig = px.scatter(
                            df, x='x', y='y',
                            hover_data=['text'],
                            title=f"Embedding Space ({method} - 2D)"
                        )
                        
                        fig.update_traces(marker=dict(size=12, color='blue'))
                        fig.update_layout(height=600, width=800)
                        
                    else:
                        df = pd.DataFrame({
                            'x': reduced[:, 0],
                            'y': reduced[:, 1],
                            'z': reduced[:, 2],
                            'text': current_texts
                        })
                        
                        fig = px.scatter_3d(
                            df, x='x', y='y', z='z',
                            hover_data=['text'],
                            title=f"Embedding Space ({method} - 3D)"
                        )
                        
                        fig.update_traces(marker=dict(size=8, color='blue'))
                        fig.update_layout(height=600)
                    
                    st.plotly_chart(fig, use_container_width=True, key="scatter_plot")
                    
                except Exception as e:
                    st.error(f"Plot error: {str(e)}")
            else:
                st.info("Click 'Generate Visualization' to see the plot")

with tab4:
    st.header("RAG Pipeline with LangChain")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        sample_doc = """Artificial Intelligence (AI) is transforming the world in unprecedented ways. 
Machine learning, a subset of AI, enables computers to learn from data without explicit programming. 
Deep learning uses neural networks with multiple layers to process complex patterns.

Natural Language Processing (NLP) allows machines to understand and generate human language. 
Recent advances in transformer models like GPT and BERT have revolutionized NLP tasks.

Computer vision enables machines to interpret and understand visual information from the world. 
Applications range from facial recognition to autonomous vehicles.

The field of AI continues to evolve rapidly, with new techniques and applications emerging constantly. 
Ethical considerations and responsible AI development are becoming increasingly important."""
        
        document_text = st.text_area("Enter document text", value=sample_doc, height=300)
        
        chunk_size = st.slider("Chunk size", 100, 500, 200)
        chunk_overlap = st.slider("Chunk overlap", 0, 100, 50)
    
    with col2:
        if st.button("Process & Index Document"):
            if not api_key:
                st.error("Please set OPENAI_API_KEY in .env file!")
            else:
                with st.spinner("Processing document..."):
                    try:
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            length_function=len,
                        )
                        
                        chunks = text_splitter.split_text(document_text)
                        
                        documents = [
                            Document(page_content=chunk, metadata={"chunk": i})
                            for i, chunk in enumerate(chunks)
                        ]
                        
                        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
                        vectorstore = LangChainFAISS.from_documents(
                            documents,
                            embeddings_model
                        )
                        
                        st.session_state.rag_vectorstore = vectorstore
                        st.session_state.rag_chunks = chunks
                        
                        st.success(f"✅ Processed {len(chunks)} chunks!")
                        
                        with st.expander("View Chunks"):
                            for i, chunk in enumerate(chunks):
                                st.markdown(f"**Chunk {i+1}:**")
                                st.text(chunk)
                                st.markdown("---")
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    
    if 'rag_vectorstore' in st.session_state:
        query = st.text_input("Ask a question", placeholder="e.g., 'What is machine learning?'")
        
        retrieve_k = st.slider("Chunks to retrieve", 1, 5, 3)
        
        if query and st.button("Get Answer", type="primary"):
            with st.spinner("Generating answer..."):
                try:
                    docs = st.session_state.rag_vectorstore.similarity_search(query, k=retrieve_k)
                    
                    st.subheader("Retrieved Context")
                    for i, doc in enumerate(docs, 1):
                        with st.expander(f"Chunk {i}"):
                            st.write(doc.page_content)
                    
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
                    
                    response = openai.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=300
                    )
                    
                    answer = response.choices[0].message.content
                    
                    st.subheader("Answer")
                    st.success(answer)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.info("Process a document first")

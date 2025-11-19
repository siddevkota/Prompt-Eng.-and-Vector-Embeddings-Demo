"""
Utility functions for the embeddings demo
"""

import numpy as np
from typing import List, Tuple
import plotly.graph_objects as go


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score (0-1)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix
    
    Args:
        embeddings: Array of embeddings (n_samples, n_features)
        
    Returns:
        Similarity matrix (n_samples, n_samples)
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    
    # Compute similarity matrix
    similarity_matrix = np.dot(normalized, normalized.T)
    return similarity_matrix


def get_top_k_similar(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    k: int = 5
) -> List[Tuple[int, float]]:
    """
    Find top-k most similar embeddings to query
    
    Args:
        query_embedding: Query vector
        embeddings: Array of embeddings to search
        k: Number of results to return
        
    Returns:
        List of (index, similarity_score) tuples
    """
    similarities = []
    for i, emb in enumerate(embeddings):
        sim = compute_cosine_similarity(query_embedding, emb)
        similarities.append((i, sim))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:k]


def create_embedding_stats(embeddings: np.ndarray) -> dict:
    """
    Compute statistics about embeddings
    
    Args:
        embeddings: Array of embeddings
        
    Returns:
        Dictionary of statistics
    """
    return {
        'n_samples': embeddings.shape[0],
        'n_dimensions': embeddings.shape[1],
        'mean': np.mean(embeddings),
        'std': np.std(embeddings),
        'min': np.min(embeddings),
        'max': np.max(embeddings),
        'norm_mean': np.mean(np.linalg.norm(embeddings, axis=1)),
        'norm_std': np.std(np.linalg.norm(embeddings, axis=1))
    }


def format_embedding_display(embedding: np.ndarray, n_dims: int = 10) -> str:
    """
    Format embedding for display
    
    Args:
        embedding: Embedding vector
        n_dims: Number of dimensions to show
        
    Returns:
        Formatted string
    """
    shown = embedding[:n_dims]
    shown_str = ', '.join([f'{x:.4f}' for x in shown])
    
    if len(embedding) > n_dims:
        shown_str += f', ... ({len(embedding) - n_dims} more)'
    
    return f'[{shown_str}]'


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """
    Simple text chunking with overlap
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence or word boundary
        if end < len(text):
            # Look for sentence end
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            
            if last_period > chunk_size * 0.5:
                chunk = chunk[:last_period + 1]
                end = start + last_period + 1
            elif last_newline > chunk_size * 0.5:
                chunk = chunk[:last_newline + 1]
                end = start + last_newline + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return [c for c in chunks if c]  # Remove empty chunks


def highlight_text(text: str, query: str) -> str:
    """
    Highlight query terms in text (simple version)
    
    Args:
        text: Text to highlight in
        query: Query to highlight
        
    Returns:
        HTML string with highlights
    """
    # Simple case-insensitive highlighting
    words = query.lower().split()
    result = text
    
    for word in words:
        # Simple replacement (not perfect but works for demo)
        result = result.replace(word, f'**{word}**')
        result = result.replace(word.capitalize(), f'**{word.capitalize()}**')
    
    return result


def get_embedding_model_info(model_name: str = "text-embedding-3-small") -> dict:
    """
    Get information about OpenAI embedding models
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with model info
    """
    models = {
        "text-embedding-3-small": {
            "dimensions": 1536,
            "max_tokens": 8191,
            "description": "Latest small embedding model, efficient and performant"
        },
        "text-embedding-3-large": {
            "dimensions": 3072,
            "max_tokens": 8191,
            "description": "Latest large embedding model, highest quality"
        },
        "text-embedding-ada-002": {
            "dimensions": 1536,
            "max_tokens": 8191,
            "description": "Previous generation, still widely used"
        }
    }
    
    return models.get(model_name, {"dimensions": "Unknown", "description": "Unknown model"})


def create_network_graph(
    embeddings: np.ndarray,
    texts: List[str],
    threshold: float = 0.8
) -> go.Figure:
    """
    Create a network graph showing connections between similar texts
    
    Args:
        embeddings: Array of embeddings
        texts: List of text strings
        threshold: Similarity threshold for drawing edges
        
    Returns:
        Plotly figure
    """
    # Compute similarity matrix
    sim_matrix = compute_similarity_matrix(embeddings)
    
    # Use simple 2D layout (could use graph layout algorithms)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    
    # Create edges
    edge_x = []
    edge_y = []
    
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if sim_matrix[i, j] > threshold:
                edge_x.extend([coords[i, 0], coords[j, 0], None])
                edge_y.extend([coords[i, 1], coords[j, 1], None])
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node trace
    node_trace = go.Scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode='markers+text',
        hoverinfo='text',
        text=[f'T{i+1}' for i in range(len(texts))],
        hovertext=texts,
        marker=dict(
            size=20,
            color='lightblue',
            line=dict(width=2, color='darkblue')
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=f'Text Similarity Network (threshold={threshold})',
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig


# Color schemes for visualizations
COLOR_SCHEMES = {
    'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'viridis': ['#440154', '#31688e', '#35b779', '#fde724'],
    'warm': ['#d62728', '#ff7f0e', '#fde724', '#ffffbf'],
    'cool': ['#1f77b4', '#6baed6', '#c6dbef', '#eff3ff']
}


def get_color_scheme(name: str = 'default') -> List[str]:
    """Get color scheme for visualizations"""
    return COLOR_SCHEMES.get(name, COLOR_SCHEMES['default'])

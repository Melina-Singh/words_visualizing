import plotly.graph_objects as go
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from embeddings import TFIDFEmbeddings
from logger import setup_logger
import json

# Initialize logger
logger = setup_logger('visualize')

def generate_visualization_data(embeddings, vocab, method='pca', top_n=20):
    logger.info(f"Generating visualization data for {method.upper()}")
    try:
        # Get reduced embeddings
        reduced_3d = embeddings.reduce_dimensions_3d(method=method)
        reduced_2d = reduced_3d[:, :2]  # Use first 2 dims for 2D
        
        # Select top N words by TF-IDF frequency (sum of scores across documents)
        word_scores = np.sum(embeddings.embedding_matrix, axis=0)
        top_indices = np.argsort(word_scores)[-top_n:][::-1]
        top_words = [vocab[i] for i in top_indices]
        
        # Compute cosine similarity matrix for top N words
        top_embeddings = embeddings.embedding_matrix[:, top_indices]
        similarity_matrix = cosine_similarity(top_embeddings.T)
        
        # 3D Scatter Plot
        scatter_3d_trace = {
            'type': 'scatter3d',
            'x': reduced_3d[:, 0].tolist(),
            'y': reduced_3d[:, 1].tolist(),
            'z': reduced_3d[:, 2].tolist(),
            'mode': 'markers+text',
            'text': vocab,
            'marker': {
                'size': 5,
                'color': 'blue',
                'opacity': 0.5
            },
            'textposition': 'top center'
        }
        scatter_3d_layout = {
            'title': f'3D Word Embeddings ({method.upper()})',
            'scene': {
                'xaxis': {'title': 'Component 1'},
                'yaxis': {'title': 'Component 2'},
                'zaxis': {'title': 'Component 3'}
            },
            'margin': {'l': 0, 'r': 0, 'b': 0, 't': 50}
        }
        
        # 2D Scatter Plot
        scatter_2d_trace = {
            'type': 'scatter',
            'x': reduced_2d[:, 0].tolist(),
            'y': reduced_2d[:, 1].tolist(),
            'mode': 'markers+text',
            'text': vocab,
            'marker': {
                'size': 10,
                'color': 'blue',
                'opacity': 0.5
            },
            'textposition': 'top center'
        }
        scatter_2d_layout = {
            'title': f'2D Word Embeddings ({method.upper()})',
            'xaxis': {'title': 'Component 1'},
            'yaxis': {'title': 'Component 2'},
            'margin': {'l': 50, 'r': 50, 'b': 50, 't': 50}
        }
        
        # Heatmap
        heatmap_trace = {
            'type': 'heatmap',
            'z': similarity_matrix.tolist(),
            'x': top_words,
            'y': top_words,
            'colorscale': 'Viridis',
            'showscale': True
        }
        heatmap_layout = {
            'title': 'Cosine Similarity Heatmap (Top 20 Words)',
            'xaxis': {'tickangle': 45},
            'yaxis': {'tickangle': 45},
            'margin': {'l': 100, 'r': 50, 'b': 150, 't': 50}
        }
        
        # Bar Chart (placeholder, updated per word in app.py)
        bar_trace = {
            'type': 'bar',
            'x': [],
            'y': [],
            'marker': {'color': 'blue'}
        }
        bar_layout = {
            'title': 'Nearest Neighbor Similarities',
            'xaxis': {'title': 'Words'},
            'yaxis': {'title': 'Cosine Similarity', 'range': [0, 1]},
            'margin': {'l': 50, 'r': 50, 'b': 100, 't': 50}
        }
        
        visualization_data = {
            'scatter_3d': {'data': [scatter_3d_trace], 'layout': scatter_3d_layout},
            'scatter_2d': {'data': [scatter_2d_trace], 'layout': scatter_2d_layout},
            'heatmap': {'data': [heatmap_trace], 'layout': heatmap_layout},
            'bar': {'data': [bar_trace], 'layout': bar_layout}
        }
        
        # Save as JSON for frontend
        with open(f'static/embeddings_vis_{method}.json', 'w') as f:
            json.dump(visualization_data, f)
        logger.info(f"Visualization data saved as JSON for {method.upper()}")
        
        return visualization_data
    except Exception as e:
        logger.error(f"Failed to generate visualization for {method.upper()}: {str(e)}")
        raise
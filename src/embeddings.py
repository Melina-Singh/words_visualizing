import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from logger import setup_logger
import os

# Silence joblib UserWarning for physical cores detection
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Adjust based on your CPU

# Initialize logger
logger = setup_logger('embeddings')

class TFIDFEmbeddings:
    def __init__(self, corpus, vocab):
        self.corpus = corpus
        self.vocab = vocab
        self.embedding_dim = None
        self.embeddings = {}
        self.embedding_matrix = None
        self.vectorizer = TfidfVectorizer(vocabulary=vocab, lowercase=True)
        logger.info(f"Initialized TF-IDF embeddings with vocab size {len(vocab)}")
        
    def compute_tfidf(self):
        logger.info("Computing TF-IDF vectors")
        try:
            tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
            self.embedding_matrix = tfidf_matrix.toarray()
            self.embedding_dim = self.embedding_matrix.shape[1]
            
            for i, word in enumerate(self.vocab):
                self.embeddings[word] = self.embedding_matrix[:, i]
            
            logger.info(f"Computed TF-IDF vectors for {len(self.corpus)} documents")
        except Exception as e:
            logger.error(f"Failed to compute TF-IDF vectors: {str(e)}")
            raise
    
    def get_nearest_neighbors(self, word, k=5):
        if word not in self.embeddings:
            logger.warning(f"Word '{word}' not in vocabulary")
            return []
        
        logger.debug(f"Computing nearest neighbors for '{word}'")
        word_vector = self.embeddings[word].reshape(1, -1)
        similarities = {}
        for other_word in self.embeddings:
            if other_word != word:
                other_vector = self.embeddings[other_word].reshape(1, -1)
                sim = cosine_similarity(word_vector, other_vector)[0][0]
                similarities[other_word] = sim
        
        neighbors = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
        logger.info(f"Found {len(neighbors)} nearest neighbors for '{word}'")
        return neighbors
    
    def reduce_dimensions_3d(self, method='pca'):
        logger.info(f"Reducing dimensions to 3D with {method.upper()}")
        try:
            # Transpose matrix to have shape (num_vocab, num_documents)
            matrix = self.embedding_matrix.T
            if method.lower() == 'pca':
                reducer = PCA(n_components=3)
                reduced_embeddings = reducer.fit_transform(matrix)
            elif method.lower() == 'tsne':
                reducer = TSNE(n_components=3, random_state=42, perplexity=min(5, len(self.vocab)-1))
                reduced_embeddings = reducer.fit_transform(matrix)
            else:
                raise ValueError(f"Unsupported reduction method: {method}")
            logger.info(f"3D dimensionality reduction complete with {method.upper()}")
            return reduced_embeddings
        except Exception as e:
            logger.error(f"Failed to reduce dimensions with {method.upper()}: {str(e)}")
            raise
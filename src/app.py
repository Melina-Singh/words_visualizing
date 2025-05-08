from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from embeddings import TFIDFEmbeddings
from preprocess import preprocess_corpus
from visualization import generate_visualization_data
from logger import setup_logger

app = Flask(__name__)
CORS(app)

# Initialize logger
logger = setup_logger('api')

# Initialize embeddings
logger.info("Initializing API with sample corpus")
sample_corpus = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is fascinating and complex",
    "Natural language processing involves analyzing text data",
    "Neural networks power modern artificial intelligence",
    "Horses gallop gracefully in the open field",
    "How to train a neural network effectively",
    "Artificial intelligence transforms modern technology",
    "Dogs and cats are popular household pets",
    "Programming in Python is both fun and powerful",
    "Deep learning models require large datasets"
]
vocab, _ = preprocess_corpus(sample_corpus, keep_stopwords=True)
try:
    embeddings = TFIDFEmbeddings(sample_corpus, vocab)
    embeddings.compute_tfidf()
    pca_vis_data = generate_visualization_data(embeddings, vocab, method='pca')
    tsne_vis_data = generate_visualization_data(embeddings, vocab, method='tsne')
except Exception as e:
    logger.error(f"Failed to initialize embeddings: {str(e)}")
    raise

@app.route('/')
def serve_frontend():
    logger.info("Serving frontend")
    return send_from_directory('../static', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    logger.info(f"Serving static file: {path}")
    return send_from_directory('../static', path)

@app.route('/embeddings', methods=['POST'])
def get_embeddings():
    logger.info("Received embeddings request")
    try:
        data = request.get_json()
        word = data.get('word', '').strip()
        method = data.get('method', 'pca').lower()
        
        if not word:
            logger.warning("Empty word received in request")
            return jsonify({'error': 'No word provided'}), 400
        
        if len(word.split()) > 1:
            logger.warning(f"Invalid input: '{word}' is not a single word")
            return jsonify({'error': f"Please enter a single word (e.g., 'neural', 'machine'). Phrases are not supported."}), 400
        
        if method not in ['pca', 'tsne']:
            logger.warning(f"Invalid reduction method: {method}")
            return jsonify({'error': f"Invalid method '{method}'. Use 'pca' or 'tsne'."}), 400
        
        if word not in embeddings.embeddings:
            logger.warning(f"Word '{word}' not in vocabulary")
            sample_words = vocab[:5] if len(vocab) >= 5 else vocab
            return jsonify({'error': f"Word '{word}' not in vocabulary. Try words like {', '.join(sample_words)}."}), 404
        
        embedding = embeddings.embeddings[word].tolist()
        neighbors = embeddings.get_nearest_neighbors(word)
        reduced_3d = embeddings.reduce_dimensions_3d(method=method)
        reduced_2d = reduced_3d[:, :2]
        word_index = embeddings.vocab.index(word) if word in embeddings.vocab else -1
        word_3d_coords = reduced_3d[word_index].tolist() if word_index >= 0 and word_index < len(reduced_3d) else []
        word_2d_coords = reduced_2d[word_index].tolist() if word_index >= 0 and word_index < len(reduced_2d) else []
        
        vis_data = pca_vis_data if method == 'pca' else tsne_vis_data
        
        bar_data = vis_data['bar']['data'][0].copy()
        bar_data['x'] = [n[0] for n in neighbors]
        bar_data['y'] = [n[1] for n in neighbors]
        vis_data['bar']['data'] = [bar_data]
        
        logger.info(f"Returning embeddings, neighbors, and visualization data for '{word}' (method: {method})")
        return jsonify({
            'word': word,
            'embedding': embedding,
            'nearest_neighbors': neighbors,
            '3d_coords': word_3d_coords,
            '2d_coords': word_2d_coords,
            'visualization_data': vis_data,
            'method': method
        })
    except Exception as e:
        logger.error(f"Error processing embeddings request: {str(e)}")
        return jsonify({'error': f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    logger.info("Starting Flask server")
    app.run(debug=True, host='0.0.0.0', port=5000)
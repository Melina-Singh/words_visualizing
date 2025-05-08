import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import string
from logger import setup_logger

# Initialize logger
logger = setup_logger('preprocess')

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_corpus(texts, keep_stopwords=False):
    logger.info("Starting corpus preprocessing")
    stop_words = set() if keep_stopwords else set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    
    processed_words = []
    for text in texts:
        logger.debug(f"Processing text: {text[:50]}...")
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens 
                 if token not in stop_words 
                 and token not in punctuation
                 and token.isalpha()]
        processed_words.extend(tokens)
    
    # Keep only top 1000 most common words
    word_freq = Counter(processed_words)
    vocab = [word for word, _ in word_freq.most_common(1000)]
    logger.info(f"Created vocabulary with {len(vocab)} words")
    return vocab, processed_words

# Example usage
if __name__ == "__main__":
    sample_corpus = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is fascinating and complex",
        "Natural language processing involves analyzing text data"
    ]
    vocab, processed = preprocess_corpus(sample_corpus, keep_stopwords=True)
    logger.info("Preprocessing complete")
    print("Vocabulary size:", len(vocab))
    print("First 10 words:", vocab[:10])
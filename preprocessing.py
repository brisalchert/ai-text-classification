from tokenizer import get_tokenizer
from nltk import PorterStemmer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

class EssayPreprocessor:
    def __init__(self, vocab):
        self.tokenizer = get_tokenizer()
        self.vocab = vocab
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def generate_tokens(self, essay):
        # Tokenize the essay
        tokens = self.tokenizer(essay)
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        # Stem remaining words
        tokens = [self.stemmer.stem(token) for token in tokens]
        return tokens

    def essay_processing_pipeline(self, essay):
        # Generate tokens
        tokens = self.generate_tokens(essay)
        # Map tokens to indices for embedding
        indices = self.vocab.map_tokens_to_index(tokens)
        return indices
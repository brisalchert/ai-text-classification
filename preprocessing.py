from tokenizer import get_tokenizer
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer

class EssayPreprocessor:
    def __init__(self, vocab=None):
        self.tokenizer = get_tokenizer()
        self.vocab = vocab
        # Only download nltk packages if necessary
        if vocab is not None:
            nltk.download("stopwords")
            nltk.download("wordnet")
        self.lemma = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self.huggingface_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def generate_tokens(self, essay):
        # Tokenize the essay
        tokens = self.tokenizer(essay)
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        # Lemmatize remaining words
        tokens = [self.lemma.lemmatize(token) for token in tokens]
        return tokens

    def essay_processing_pipeline(self, essay):
        if self.vocab is None:
            raise NameError("Preprocessor must be initialized with a vocab to use this function")
        # Generate tokens
        tokens = self.generate_tokens(essay)
        # Map tokens to indices for embedding
        indices = self.vocab.map_tokens_to_index(tokens)
        return indices

    def huggingface_pipeline(self, essay):
        tokens = self.huggingface_tokenizer.tokenize(essay)
        indices = self.huggingface_tokenizer.convert_tokens_to_ids(tokens)
        return indices[:512]

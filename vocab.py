from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
from tqdm import tqdm
from tokenizer import get_tokenizer

class VocabGenerator:
    def __init__(self, essays=None, vocab=None):
        self.tokenizer = get_tokenizer()
        self.stop_words = stopwords.words('english')
        self.stemmer = PorterStemmer()
        if essays is not None:
            self.vocab = None
            self.generate_vocab(essays)
        else:
            self.vocab = vocab

    def generate_vocab(self, essays):
        vocab = set()
        all_tokens = []
        print("Generating vocabulary:")
        for essay in tqdm(essays):
            # Tokenize the essay
            tokens = self.tokenizer(essay)
            # Add new tokens to the vocab and the list of all tokens
            for token in tokens:
                vocab.add(token)
                all_tokens.append(token)
        # Remove rare tokens from vocab
        freq_dist = FreqDist(all_tokens)
        threshold = 16
        vocab = [token for token in vocab if freq_dist[token] > threshold]
        # Remove stopwords
        vocab = [token for token in vocab if token not in self.stop_words]
        # Stem remaining tokens
        vocab = [self.stemmer.stem(token) for token in vocab]
        # Remove duplicate tokens by converting back to set
        vocab = set(vocab)
        # Convert vocab list to index mapping
        vocab_to_idx = {word: i for i, word in enumerate(vocab)}
        self.vocab = vocab_to_idx

    def map_tokens_to_index(self, tokens):
        essay_indices = []
        # Map the tokens to indices using the vocab dictionary
        for token in tokens:
            # Ignore out of vocabulary tokens
            if token in self.vocab.keys():
                essay_indices.append(self.vocab[token])
        return essay_indices

    def get_vocab_size(self):
        return len(self.vocab)

    def get_vocab_dictionary(self):
        return self.vocab
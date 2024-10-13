import torch
import pickle
import pandas as pd
from tqdm import tqdm
from essayLSTM import EssayLSTM
from preprocessing import EssayPreprocessor
from vocab import VocabGenerator

# Load dataset
ai_human_df = pd.read_csv('AI_Human.csv')

# Load vocabulary
with open('vocab.pkl', 'rb') as f:
    vocab_dict = pickle.load(f)

vocab = VocabGenerator(vocab=vocab_dict)

# Initialize essay preprocessor
preprocessor = EssayPreprocessor(vocab)

# Load model parameters
with open('model-params.pkl', 'rb') as f:
    model_params = pickle.load(f)

vocab_size = model_params['vocab_size']
embed_size = model_params['embed_size']
hidden_size = model_params['hidden_size']
num_layers = model_params['num_layers']
num_class = model_params['num_class']

# Set device to CPU
device = 'cpu'

# Load the model
model = EssayLSTM(vocab_size, embed_size, hidden_size, num_layers, num_class, device)
model.load_state_dict(torch.load('ai-text-model.pt', weights_only=True))
model.eval()
model.to('cpu')

def predict(essay, essay_pipeline):
    with torch.no_grad():
        text = essay_pipeline(essay)
        text = torch.tensor(text)
        sequence_length = torch.tensor(text.shape)
        # Reshape text for batch-first tensor format
        text = text.reshape(1, -1)
        output = model(text, sequence_length)
        return output.argmax(1).item()

gen_essay_label = {0: 'human-generated', 1: 'AI-generated'}

sample_essays = ai_human_df.sample(n=1000)
count_correct = 0
count_total = len(sample_essays)

print("Testing random sample:")
for i in tqdm(range(len(sample_essays))):
    true_label = gen_essay_label[sample_essays.iloc[i, 1]]
    pred_label = gen_essay_label[predict(sample_essays.iloc[i,0], preprocessor.essay_processing_pipeline)]

    if true_label == pred_label:
        count_correct += 1

valid_accuracy = count_correct / count_total

print('\nValidation accuracy on random sample: {:8.3f}'.format(valid_accuracy))
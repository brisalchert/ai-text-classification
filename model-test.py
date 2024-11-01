import numpy as np
import torch
import pickle
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm
from essayLSTM import EssayLSTM
from preprocessing import EssayPreprocessor
from vocab import VocabGenerator

# Load dataset
ai_human_df = pd.read_json('dataset.jsonl', lines=True)
ai_human_df = ai_human_df[["Answer", "Is_it_AI"]]

# Remove entries with missing answers
ai_human_df.dropna(subset=['Answer'], inplace=True)

print(ai_human_df.head())
print(ai_human_df.describe())
print(ai_human_df.info())

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

# Set device to CPU
device = 'cpu'

# Set prediction threshold
threshold = 0.999849

# Load the model
model = EssayLSTM(vocab_size, embed_size, hidden_size, num_layers, device)
model.load_state_dict(torch.load('ai-text-model.pt', weights_only=True))
model.eval()
model.to('cpu')

def predict(essay, essay_pipeline):
    with torch.no_grad():
        text = essay_pipeline(essay)
        text = torch.tensor(text)

        if len(text) == 0:
            raise ValueError("Invalid input for prediction.")

        sequence_length = torch.tensor(text.shape)
        # Reshape text for batch-first tensor format
        text = text.reshape(1, -1)
        output = model(text, sequence_length)
        return output

gen_essay_label = {0: 'human-generated', 1: 'AI-generated'}

sample_essays = ai_human_df
true_pos, true_neg = 0, 0
count_human, count_ai = 0, 0
count_total = len(sample_essays)

labels = []
pred_labels = []

for i in range(len(sample_essays)):
    try:
        pred_label = torch.sigmoid(predict(sample_essays.iloc[i,0], preprocessor.essay_processing_pipeline)).item()
        pred_labels.append(pred_label)

        label = sample_essays.iloc[i, 1]
        labels.append(label)
    # Ignore samples that are invalid for predictions
    except ValueError:
        pass

np.array(labels)
np.array(pred_labels)
fpr, tpr, thresholds = roc_curve(labels, pred_labels)
au_roc = roc_auc_score(labels, pred_labels)

roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
roc_df['cutoff'] = tpr - fpr

print(roc_df.sort_values("cutoff", ascending=False).head())
print("AU ROC score: ", au_roc)

plt.plot(fpr, tpr, marker='.')
plt.title("ROC Curve for Testing Data")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

for i in tqdm(range(len(sample_essays))):
    try:
        true_label = gen_essay_label[sample_essays.iloc[i, 1]]
        pred_label = gen_essay_label[(torch.sigmoid(
            predict(sample_essays.iloc[i,0],preprocessor.essay_processing_pipeline)
        ) >= threshold).int().item()]

        if true_label == pred_label:
            if true_label == 'AI-generated':
                true_pos += 1
            else:
                true_neg += 1

        if pred_label == gen_essay_label[0]:
            count_human += 1
        else:
            count_ai += 1
    except ValueError:
        pass

count_correct = true_pos + true_neg
valid_accuracy = count_correct / count_total

print('\nValidation accuracy on random sample: {:8.3f}'.format(valid_accuracy))
print('\nHuman predictions: {:5d}\nAI predictions: {:5d}'.format(count_human, count_ai))
print('\nTrue negatives: {:5d}\nTrue positives: {:5d}'.format(true_neg, true_pos))
print('\nHuman accuracy: {:8.3f}\nAI accuracy: {:8.3f}'.format((true_neg / count_human), (true_pos / count_ai)))
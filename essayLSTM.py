import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class EssayLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, device):
        super(EssayLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=False, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc_2 = nn.Linear(128, 1)
        self.device = device

    def forward(self, sequences, lengths):
        # Perform embedding
        embedded = self.embedding(sequences)
        h0 = torch.zeros(self.num_layers, embedded.size(0), self.hidden_size, device=self.device)
        c0 = torch.zeros(self.num_layers, embedded.size(0), self.hidden_size, device=self.device)
        # Pack the embedded sequences
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        # Propagate embeddings through LSTM layer
        out, (hn, cn) = self.lstm(packed, (h0, c0))
        out = hn[-1] # Get last hidden state
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc_2(out)
        return out
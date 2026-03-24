import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(AttentionLSTM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x: [batch, seq_len]
        embeds = self.embeddings(x) # [batch, seq_len, embed_dim]

        lstm_out, (h_n, c_n) = self.lstm(embeds) # lstm_out: [batch, seq_len, hidden_dim]

        # Parameter-free attention:
        # Use the mean hidden activation per timestep to create attention scores.
        # (The saved state dict does not include attention layer parameters.)
        attn_scores = lstm_out.mean(dim=2)  # [batch, seq_len]
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # [batch, seq_len, 1]

        # Context vector (weighted sum of all hidden states)
        context = torch.sum(attn_weights * lstm_out, dim=1) # [batch, hidden_dim]

        # This context vector is the "STATE" Member 4 needs!
        out = self.fc(self.dropout(context))
        return out
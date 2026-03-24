import json
import torch
import numpy as np
# Allow running both as a package (src.extract_states) and as a script.
try:
    from .model import AttentionLSTM
except ImportError:
    from model import AttentionLSTM


def get_user_state(model, sequence, device):
    """
    Takes a sequence of movie indices and returns the 128-dim
    vector (context) from the Attention layer.
    """
    model.eval()
    with torch.no_grad():
        x = torch.LongTensor([sequence]).to(device)

        # Reuse the model internals to compute the attention context vector.
        embeds = model.embeddings(x)
        lstm_out, _ = model.lstm(embeds)
        # Parameter-free attention must match src/model.py
        attn_scores = lstm_out.mean(dim=2)  # [batch, seq_len]
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # [batch, seq_len, 1]
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [batch, hidden_dim]

        return context.squeeze(0).cpu().numpy()


def load_mapping(mapping_path='data/movie2idx.json'):
    with open(mapping_path, 'r') as f:
        raw = json.load(f)
    # JSON keys are strings; convert back to ints for movieId lookups.
    return {int(k): int(v) for k, v in raw.items()}


def load_trained_model(weights_path, vocab_size, embedding_dim, hidden_dim, output_dim, device):
    model = AttentionLSTM(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('State extractor ready. Use get_user_state(model, sequence, device).')

import torch
import torch.nn as nn


class CharRNN(nn.Module):
    """A manually parameterized character-level RNN baseline."""

    def __init__(self, vocab_size: int, hidden_size: int = 256, embedding_dim: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.W_xh = nn.Parameter(torch.randn(embedding_dim, hidden_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        self.W_hy = nn.Parameter(torch.randn(hidden_size, vocab_size) * 0.01)
        self.b_y = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(x).transpose(0, 1)
        batch_size = x.size(0)

        if hidden is None or hidden.size(0) != batch_size:
            h_prev = self.init_hidden(batch_size, x.device)
        else:
            h_prev = hidden

        outputs = []
        for x_t in embedded:
            h_t = torch.tanh(x_t @ self.W_xh + h_prev @ self.W_hh + self.b_h)
            outputs.append(h_t)
            h_prev = h_t

        hidden_states = torch.stack(outputs).transpose(0, 1)
        logits = hidden_states @ self.W_hy + self.b_y
        return logits, h_t

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def config(self) -> dict[str, int]:
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "embedding_dim": self.embedding_dim,
        }

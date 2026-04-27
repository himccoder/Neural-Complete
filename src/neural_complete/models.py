import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def config(self) -> dict[str, int | str]:
        return {
            "model_type": "char_rnn",
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "embedding_dim": self.embedding_dim,
        }


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention used by GPT-style language models."""

    def __init__(self, embedding_dim: int, num_heads: int, block_size: int, dropout: float):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.qkv = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embedding_dim = x.shape
        q, k, v = self.qkv(x).split(embedding_dim, dim=2)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) * (self.head_dim**-0.5)
        scores = scores.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embedding_dim)
        return self.resid_dropout(self.proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, block_size: int, dropout: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embedding_dim)
        self.attn = CausalSelfAttention(embedding_dim, num_heads, block_size, dropout)
        self.ln_2 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class MiniGPT(nn.Module):
    """A compact decoder-only Transformer for character-level language modeling."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int = 128,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embedding_dim, num_heads, block_size, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, None]:
        del hidden
        _, seq_len = x.shape
        if seq_len > self.block_size:
            raise ValueError(f"Cannot forward sequence of length {seq_len}; block_size is {self.block_size}")

        positions = torch.arange(seq_len, device=x.device)
        x = self.token_embedding(x) + self.position_embedding(positions)[None, :, :]
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x), None

    def config(self) -> dict[str, int | float | str]:
        return {
            "model_type": "mini_gpt",
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from neural_complete.data import Vocabulary, load_text, make_dataloaders, split_token_ids
from neural_complete.evaluation import evaluate_loss
from neural_complete.models import CharRNN


def save_checkpoint(
    path: str,
    model: CharRNN,
    vocab: Vocabulary,
    sequence_length: int,
    metrics: dict[str, float],
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model.config(),
            "vocab_stoi": vocab.stoi,
            "sequence_length": sequence_length,
            "metrics": metrics,
        },
        checkpoint_path,
    )


def load_checkpoint(path: str, device: torch.device) -> tuple[CharRNN, Vocabulary, int, dict[str, float]]:
    checkpoint = torch.load(path, map_location=device)
    model = CharRNN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    stoi = checkpoint["vocab_stoi"]
    vocab = Vocabulary(stoi=stoi, itos={idx: ch for ch, idx in stoi.items()})
    return model, vocab, checkpoint["sequence_length"], checkpoint.get("metrics", {})


def train_char_rnn(
    data_file: str | None,
    output_path: str,
    epochs: int,
    batch_size: int,
    sequence_length: int,
    stride: int,
    hidden_size: int,
    embedding_dim: int,
    learning_rate: float,
    train_ratio: float,
    device: torch.device,
) -> dict[str, float]:
    text = load_text(data_file)
    vocab = Vocabulary.from_text(text)
    train_data, val_data = split_token_ids(vocab.encode(text), train_ratio=train_ratio)
    train_loader, val_loader = make_dataloaders(train_data, val_data, sequence_length, stride, batch_size)

    if len(train_loader) == 0:
        raise ValueError("Not enough training data for the chosen sequence_length and batch_size")

    model = CharRNN(vocab_size=vocab.size, hidden_size=hidden_size, embedding_dim=embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    metrics: dict[str, float] = {}
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for inputs, targets in progress:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            logits, _ = model(inputs)
            loss = criterion(logits.reshape(-1, vocab.size), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(train_loss=f"{loss.item():.4f}")

        train_loss = total_loss / len(train_loader)
        val_loss, val_perplexity = evaluate_loss(model, val_loader, criterion, device)
        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_perplexity": val_perplexity,
            "vocab_size": float(vocab.size),
        }
        print(
            f"epoch={epoch + 1} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_perplexity={val_perplexity:.2f}"
        )

    save_checkpoint(output_path, model, vocab, sequence_length, metrics)
    return metrics

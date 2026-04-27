import json
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from neural_complete.data import Vocabulary, load_text, make_dataloaders, split_token_ids
from neural_complete.evaluation import evaluate_loss
from neural_complete.models import CharRNN, MiniGPT


def save_checkpoint(
    path: str,
    model: CharRNN | MiniGPT,
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


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def default_metrics_path(output_path: str) -> str:
    checkpoint_path = Path(output_path)
    return str(checkpoint_path.with_suffix(".metrics.json"))


def save_metrics_json(
    path: str,
    model: CharRNN | MiniGPT,
    data_file: str | None,
    sequence_length: int,
    batch_size: int,
    stride: int,
    learning_rate: float,
    train_ratio: float,
    history: list[dict[str, float]],
    final_metrics: dict[str, float],
) -> None:
    metrics_path = Path(path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_config": model.config(),
        "parameter_count": count_parameters(model),
        "data_file": data_file or "built-in alphabet demo",
        "training_config": {
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "stride": stride,
            "learning_rate": learning_rate,
            "train_ratio": train_ratio,
        },
        "history": history,
        "final_metrics": final_metrics,
    }
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def build_model(model_config: dict, device: torch.device) -> CharRNN | MiniGPT:
    model_type = model_config.get("model_type", "char_rnn")
    config = dict(model_config)
    config.pop("model_type", None)

    if model_type == "char_rnn":
        return CharRNN(**config).to(device)
    if model_type == "mini_gpt":
        return MiniGPT(**config).to(device)
    raise ValueError(f"Unsupported model_type: {model_type}")


def load_checkpoint(path: str, device: torch.device) -> tuple[CharRNN | MiniGPT, Vocabulary, int, dict[str, float]]:
    checkpoint = torch.load(path, map_location=device)
    model = build_model(checkpoint["model_config"], device)
    model.load_state_dict(checkpoint["model_state"])
    stoi = checkpoint["vocab_stoi"]
    vocab = Vocabulary(stoi=stoi, itos={idx: ch for ch, idx in stoi.items()})
    return model, vocab, checkpoint["sequence_length"], checkpoint.get("metrics", {})


def train_language_model(
    model: CharRNN | MiniGPT,
    data_file: str | None,
    output_path: str,
    epochs: int,
    batch_size: int,
    sequence_length: int,
    stride: int,
    learning_rate: float,
    train_ratio: float,
    device: torch.device,
    metrics_output: str | None = None,
) -> dict[str, float]:
    text = load_text(data_file)
    vocab = Vocabulary.from_text(text)
    train_data, val_data = split_token_ids(vocab.encode(text), train_ratio=train_ratio)
    train_loader, val_loader = make_dataloaders(train_data, val_data, sequence_length, stride, batch_size)

    if len(train_loader) == 0:
        raise ValueError("Not enough training data for the chosen sequence_length and batch_size")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    metrics: dict[str, float] = {}
    history: list[dict[str, float]] = []
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
            "epoch": float(epoch + 1),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_perplexity": val_perplexity,
            "vocab_size": float(vocab.size),
            "parameter_count": float(count_parameters(model)),
        }
        history.append(metrics)
        print(
            f"epoch={epoch + 1} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_perplexity={val_perplexity:.2f}"
        )

    save_checkpoint(output_path, model, vocab, sequence_length, metrics)
    save_metrics_json(
        path=metrics_output or default_metrics_path(output_path),
        model=model,
        data_file=data_file,
        sequence_length=sequence_length,
        batch_size=batch_size,
        stride=stride,
        learning_rate=learning_rate,
        train_ratio=train_ratio,
        history=history,
        final_metrics=metrics,
    )
    return metrics


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
    metrics_output: str | None = None,
) -> dict[str, float]:
    text = load_text(data_file)
    vocab = Vocabulary.from_text(text)
    model = CharRNN(vocab_size=vocab.size, hidden_size=hidden_size, embedding_dim=embedding_dim)
    return train_language_model(
        model=model,
        data_file=data_file,
        output_path=output_path,
        epochs=epochs,
        batch_size=batch_size,
        sequence_length=sequence_length,
        stride=stride,
        learning_rate=learning_rate,
        train_ratio=train_ratio,
        device=device,
        metrics_output=metrics_output,
    )


def train_mini_gpt(
    data_file: str | None,
    output_path: str,
    epochs: int,
    batch_size: int,
    sequence_length: int,
    stride: int,
    embedding_dim: int,
    num_heads: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    train_ratio: float,
    device: torch.device,
    metrics_output: str | None = None,
) -> dict[str, float]:
    text = load_text(data_file)
    vocab = Vocabulary.from_text(text)
    model = MiniGPT(
        vocab_size=vocab.size,
        block_size=sequence_length,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )
    return train_language_model(
        model=model,
        data_file=data_file,
        output_path=output_path,
        epochs=epochs,
        batch_size=batch_size,
        sequence_length=sequence_length,
        stride=stride,
        learning_rate=learning_rate,
        train_ratio=train_ratio,
        device=device,
        metrics_output=metrics_output,
    )

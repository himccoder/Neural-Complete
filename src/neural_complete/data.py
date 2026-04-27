import re
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class Vocabulary:
    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> "Vocabulary":
        chars = sorted(set(text))
        stoi = {ch: idx for idx, ch in enumerate(chars)}
        itos = {idx: ch for ch, idx in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.itos[token_id] for token_id in token_ids)


class CharDataset(Dataset):
    """Overlapping next-character prediction windows."""

    def __init__(self, token_ids: torch.Tensor, sequence_length: int, stride: int):
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if stride < 1:
            raise ValueError("stride must be at least 1")

        self.inputs = []
        self.targets = []
        for i in range(0, len(token_ids) - sequence_length, stride):
            self.inputs.append(token_ids[i : i + sequence_length])
            self.targets.append(token_ids[i + 1 : i + sequence_length + 1])

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx].long(), self.targets[idx].long()


def clean_text(text: str) -> str:
    text = text.lower()
    return re.sub(r"[^a-z0-9.,!?;:()\[\]'\" \n-]+", "", text)


def load_text(path: str | None, fallback_text: str | None = None) -> str:
    if path:
        with open(path, "r", encoding="utf-8") as file:
            return clean_text(file.read())
    if fallback_text:
        return clean_text(fallback_text)
    return "abcdefghijklmnopqrstuvwxyz " * 100


def split_token_ids(token_ids: list[int], train_ratio: float = 0.9) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    data = torch.tensor(token_ids, dtype=torch.long)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]


def make_dataloaders(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    sequence_length: int,
    stride: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = CharDataset(train_data, sequence_length, stride)
    val_dataset = CharDataset(val_data, sequence_length, stride)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader

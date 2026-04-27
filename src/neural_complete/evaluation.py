import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_loss(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    batches = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits, _ = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            total_loss += loss.item()
            batches += 1

    if batches == 0:
        return float("nan"), float("nan")

    avg_loss = total_loss / batches
    return avg_loss, math.exp(avg_loss)

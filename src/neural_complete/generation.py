import torch

from neural_complete.data import Vocabulary
from neural_complete.sampling import sample_logits


def generate_text(
    model: torch.nn.Module,
    vocab: Vocabulary,
    prompt: str,
    max_new_chars: int,
    sequence_length: int,
    device: torch.device,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    greedy: bool = False,
) -> str:
    model.eval()
    generated = prompt.lower()
    token_ids = vocab.encode(generated)

    if not token_ids:
        fallback = " " if " " in vocab.stoi else next(iter(vocab.stoi))
        token_ids = [vocab.stoi[fallback]]
        generated = fallback

    pad_id = vocab.stoi.get(" ", token_ids[0])

    with torch.no_grad():
        for _ in range(max_new_chars):
            context = token_ids[-sequence_length:]
            if len(context) < sequence_length:
                context = [pad_id] * (sequence_length - len(context)) + context

            x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = model(x)

            next_id = sample_logits(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                greedy=greedy,
            ).item()
            token_ids.append(next_id)
            generated += vocab.itos[next_id]

    return generated

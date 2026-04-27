import argparse

import torch

from neural_complete.generation import generate_text
from neural_complete.training import load_checkpoint, train_char_rnn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and sample small language models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train-rnn", help="Train the scratch character RNN baseline.")
    train.add_argument("--data-file", default=None, help="Path to a text corpus. Defaults to alphabet demo data.")
    train.add_argument("--output", default="checkpoints/char_rnn.pt", help="Where to save the checkpoint.")
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--sequence-length", type=int, default=50)
    train.add_argument("--stride", type=int, default=3)
    train.add_argument("--hidden-size", type=int, default=256)
    train.add_argument("--embedding-dim", type=int, default=128)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--train-ratio", type=float, default=0.9)
    train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    generate = subparsers.add_parser("generate", help="Generate text from a saved checkpoint.")
    generate.add_argument("--checkpoint", default="checkpoints/char_rnn.pt")
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--max-new-chars", type=int, default=100)
    generate.add_argument("--temperature", type=float, default=1.0)
    generate.add_argument("--top-k", type=int, default=None)
    generate.add_argument("--top-p", type=float, default=None)
    generate.add_argument("--greedy", action="store_true")
    generate.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    device = torch.device(args.device)

    if args.command == "train-rnn":
        metrics = train_char_rnn(
            data_file=args.data_file,
            output_path=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            stride=args.stride,
            hidden_size=args.hidden_size,
            embedding_dim=args.embedding_dim,
            learning_rate=args.learning_rate,
            train_ratio=args.train_ratio,
            device=device,
        )
        print(f"saved_checkpoint={args.output}")
        print(metrics)
        return

    if args.command == "generate":
        model, vocab, sequence_length, metrics = load_checkpoint(args.checkpoint, device)
        text = generate_text(
            model=model,
            vocab=vocab,
            prompt=args.prompt,
            max_new_chars=args.max_new_chars,
            sequence_length=sequence_length,
            device=device,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            greedy=args.greedy,
        )
        print(text)
        if metrics:
            print(f"\ncheckpoint_metrics={metrics}")


if __name__ == "__main__":
    main()

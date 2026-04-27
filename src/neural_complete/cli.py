import argparse
import json
from pathlib import Path

import torch

from neural_complete.generation import generate_text
from neural_complete.training import count_parameters, load_checkpoint, train_char_rnn, train_mini_gpt


def parse_checkpoint_arg(value: str) -> tuple[str, str]:
    if "=" in value:
        name, path = value.split("=", 1)
        return name.strip(), path.strip()

    path = value.strip()
    return Path(path).stem, path


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
    train.add_argument("--metrics-output", default=None, help="Optional JSON path for training metrics.")
    train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    train_gpt = subparsers.add_parser("train-gpt", help="Train a small decoder-only Transformer.")
    train_gpt.add_argument("--data-file", default=None, help="Path to a text corpus. Defaults to alphabet demo data.")
    train_gpt.add_argument("--output", default="checkpoints/mini_gpt.pt", help="Where to save the checkpoint.")
    train_gpt.add_argument("--epochs", type=int, default=5)
    train_gpt.add_argument("--batch-size", type=int, default=64)
    train_gpt.add_argument("--sequence-length", type=int, default=64)
    train_gpt.add_argument("--stride", type=int, default=3)
    train_gpt.add_argument("--embedding-dim", type=int, default=128)
    train_gpt.add_argument("--num-heads", type=int, default=4)
    train_gpt.add_argument("--num-layers", type=int, default=4)
    train_gpt.add_argument("--dropout", type=float, default=0.1)
    train_gpt.add_argument("--learning-rate", type=float, default=3e-4)
    train_gpt.add_argument("--train-ratio", type=float, default=0.9)
    train_gpt.add_argument("--metrics-output", default=None, help="Optional JSON path for training metrics.")
    train_gpt.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    generate = subparsers.add_parser("generate", help="Generate text from a saved checkpoint.")
    generate.add_argument("--checkpoint", default="checkpoints/char_rnn.pt")
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--max-new-chars", type=int, default=100)
    generate.add_argument("--temperature", type=float, default=1.0)
    generate.add_argument("--top-k", type=int, default=None)
    generate.add_argument("--top-p", type=float, default=None)
    generate.add_argument("--greedy", action="store_true")
    generate.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    compare = subparsers.add_parser("compare", help="Compare checkpoints on metrics and generated text.")
    compare.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Checkpoint to compare. Use name=path or just path. Repeat for multiple models.",
    )
    compare.add_argument("--prompt", required=True)
    compare.add_argument("--max-new-chars", type=int, default=120)
    compare.add_argument("--temperature", type=float, default=1.0)
    compare.add_argument("--top-k", type=int, default=None)
    compare.add_argument("--top-p", type=float, default=None)
    compare.add_argument("--greedy", action="store_true")
    compare.add_argument("--output-json", default=None, help="Optional path to save comparison results.")
    compare.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

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
            metrics_output=args.metrics_output,
        )
        print(f"saved_checkpoint={args.output}")
        print(metrics)
        return

    if args.command == "train-gpt":
        metrics = train_mini_gpt(
            data_file=args.data_file,
            output_path=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            stride=args.stride,
            embedding_dim=args.embedding_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            train_ratio=args.train_ratio,
            device=device,
            metrics_output=args.metrics_output,
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

    if args.command == "compare":
        results = []
        for checkpoint_arg in args.checkpoint:
            name, path = parse_checkpoint_arg(checkpoint_arg)
            model, vocab, sequence_length, metrics = load_checkpoint(path, device)
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
            result = {
                "name": name,
                "checkpoint": path,
                "model_type": model.config().get("model_type", "unknown"),
                "parameter_count": count_parameters(model),
                "metrics": metrics,
                "generated_text": text,
            }
            results.append(result)

        for result in results:
            metrics = result["metrics"]
            print(f"\n== {result['name']} ({result['model_type']}) ==")
            print(f"checkpoint={result['checkpoint']}")
            print(f"parameters={result['parameter_count']}")
            if metrics:
                print(
                    "metrics="
                    f"train_loss={metrics.get('train_loss')}, "
                    f"val_loss={metrics.get('val_loss')}, "
                    f"val_perplexity={metrics.get('val_perplexity')}"
                )
            print(f"generated_text={result['generated_text']}")

        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump({"prompt": args.prompt, "results": results}, file, indent=2)


if __name__ == "__main__":
    main()

# Neural Complete

Neural Complete is a small language-modeling project built in PyTorch. It explores how neural networks learn text patterns and generate new text one character at a time.

The project starts with a manually implemented character-level RNN and extends it with a compact GPT-style decoder-only Transformer. Both models are trained to predict the next character in a sequence, then used for autoregressive text generation. This makes the project useful for understanding the connection between classic sequence models and modern language model ideas such as causal self-attention, positional embeddings, decoding strategies, and perplexity.

At a high level, the project answers this question:

> Given some text, can a neural network learn its character patterns well enough to continue a prompt?

The RNN baseline shows the limits of simple recurrent models, while the Mini-GPT model introduces the Transformer architecture used by modern LLMs.

## What This Project Includes

- A manually implemented character-level RNN in PyTorch.
- A compact GPT-style decoder-only Transformer implemented in PyTorch.
- Reusable dataset, vocabulary, training, evaluation, and generation modules.
- Autoregressive text completion with temperature, greedy decoding, top-k sampling, and nucleus sampling.
- Checkpoint-based training and generation.
- LLM-style evaluation using validation loss and perplexity.

## Current Structure

```text
src/neural_complete/
  cli.py          # Command-line entry point for training and generation
  data.py         # Text cleaning, vocabulary, dataset, and dataloaders
  evaluation.py   # Validation loss and perplexity
  generation.py   # Autoregressive text generation
  models.py       # Scratch character RNN and Mini-GPT Transformer
  sampling.py     # Greedy, temperature, top-k, and top-p decoding
  training.py     # Training loop and checkpoint utilities

rnn_complete.py   # Original assignment-style script
requirements.txt
```

## Setup

```bash
pip install -e .
```

You can also install only the raw dependencies:

```bash
pip install -r requirements.txt
```

## Train the RNN Baseline

Train on the built-in alphabet demo:

```bash
neural-complete train-rnn --epochs 3 --output checkpoints/char_rnn.pt
```

Train on your own corpus:

```bash
neural-complete train-rnn --data-file warandpeace.txt --epochs 5 --output checkpoints/warpeace_rnn.pt
```

The checkpoint stores the model weights, vocabulary, sequence length, and final metrics.

## Train the Mini-GPT Transformer

Train a compact decoder-only Transformer on the built-in alphabet demo:

```bash
neural-complete train-gpt --epochs 5 --output checkpoints/mini_gpt.pt
```

Train on your own corpus:

```bash
neural-complete train-gpt \
  --data-file warandpeace.txt \
  --epochs 5 \
  --sequence-length 128 \
  --embedding-dim 128 \
  --num-heads 4 \
  --num-layers 4 \
  --output checkpoints/warpeace_gpt.pt
```

The Mini-GPT model includes:

- Token embeddings.
- Learned positional embeddings.
- Multi-head causal self-attention.
- Feed-forward Transformer blocks.
- Residual connections.
- LayerNorm.
- Autoregressive generation.

## Generate Text

```bash
neural-complete generate --checkpoint checkpoints/char_rnn.pt --prompt "abc" --max-new-chars 80
```

Use LLM decoding controls:

```bash
neural-complete generate \
  --checkpoint checkpoints/warpeace_rnn.pt \
  --prompt "war" \
  --max-new-chars 200 \
  --temperature 0.8 \
  --top-k 20 \
  --top-p 0.9
```

The same generation command works for RNN and Mini-GPT checkpoints because both expose the same language-modeling interface.

## Next Steps

- Compare RNN and Mini-GPT results on the same text corpus.
- Add plots for training loss, validation loss, and perplexity.
- Save experiment metrics to JSON for easier comparison.
- Add attention visualization for the Mini-GPT model.
- Add subword tokenization experiments.

## Original Experiment Summary

The initial version trained a manually implemented character RNN on a repeated alphabet sequence and on War and Peace. The alphabet task reached very low loss and generated near-perfect continuations. The War and Peace experiment produced English-like fragments but showed the limitations of a simple RNN on long-range natural language structure, motivating the next step: Transformer-based language modeling.

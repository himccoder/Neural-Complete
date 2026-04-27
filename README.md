# Neural Complete

Neural Complete is evolving from a character-level RNN assignment into a compact language-modeling portfolio project. The goal is to show the path from classic sequence modeling to modern LLM concepts: tokenization, autoregressive generation, decoding strategies, evaluation, Transformers, and eventually retrieval-augmented generation.

## What This Project Demonstrates

- A manually implemented character-level RNN in PyTorch.
- A compact GPT-style decoder-only Transformer implemented in PyTorch.
- Reusable dataset, vocabulary, training, evaluation, and generation modules.
- Autoregressive text completion with temperature, greedy decoding, top-k sampling, and nucleus sampling.
- Checkpoint-based training and generation.
- LLM-style evaluation using validation loss and perplexity.
- A clear roadmap toward tokenizer experiments, model comparison, and RAG.

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

## Roadmap Toward a Strong CV Project

### Phase 1: Productionize the RNN Baseline

This phase is now started. The project has reusable modules, a CLI, checkpoints, decoding controls, and perplexity evaluation.

Useful next improvements:

- Add plots for train/validation loss.
- Save metrics to JSON.
- Add unit tests for tokenization, sampling, and generation.
- Add experiment configs for different corpora.

### Phase 2: Expand the Mini-GPT Transformer

This phase is now started. The project has a decoder-only Transformer with:

- Token embeddings and positional embeddings.
- Multi-head causal self-attention.
- Feed-forward blocks.
- Residual connections and LayerNorm.
- Autoregressive generation with the same decoding controls.

Useful next improvements:

- Add attention visualization.
- Add model parameter counts and training-time comparisons.
- Add a stronger experiment on a real corpus.
- Add a writeup comparing RNN limitations with Transformer attention.

### Phase 3: Add Tokenization Experiments

Compare character-level modeling with subword tokenization:

- Character tokenizer baseline.
- Byte-pair encoding or Hugging Face tokenizer.
- Vocabulary size comparison.
- Sequence length and perplexity comparison.

### Phase 4: Add Model Comparison

Compare the original scratch RNN, a stronger recurrent baseline such as GRU or LSTM, and the Mini-GPT model.

Track:

- Validation loss.
- Perplexity.
- Sample quality.
- Repetition rate.
- Training time.

### Phase 5: Add RAG

Build a retrieval-augmented generation demo over a text corpus:

- Chunk source documents.
- Embed chunks with a sentence-transformer model.
- Store embeddings in FAISS or Chroma.
- Retrieve relevant chunks for a query.
- Construct a grounded answer with citations.

This adds applied LLM engineering concepts beyond model training.

### Phase 6: Add a Demo App

Use Streamlit or FastAPI to expose:

- Text completion.
- Sampling controls.
- Model comparison.
- RAG question answering.
- Training metrics.

## CV Description

Suggested CV bullet:

> Built an end-to-end PyTorch language-modeling project evolving from a scratch character RNN to a compact GPT-style Transformer, including causal self-attention, autoregressive decoding, top-k/top-p sampling, perplexity evaluation, checkpointing, and a roadmap toward tokenization experiments and RAG.

## Original Experiment Summary

The initial version trained a manually implemented character RNN on a repeated alphabet sequence and on War and Peace. The alphabet task reached very low loss and generated near-perfect continuations. The War and Peace experiment produced English-like fragments but showed the limitations of a simple RNN on long-range natural language structure, motivating the next step: Transformer-based language modeling.

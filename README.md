# My GPT — Built from Scratch

> Assembled from the NeetCode ML course on [NeetCode.io](https://neetcode.io)
> Built by **Nithisha Yasam** on August 29, 2026

Every file in this project is code I wrote and submitted while completing the NeetCode ML course.
The problems progressively build from gradient descent fundamentals all the way to a working GPT.

## Project Structure

```
model/          Attention, Transformer, GPT architecture
  attention.py             Self-attention head
  multi_head_attention.py  Multi-headed attention
  transformer.py           Transformer block
  gpt.py                   GPT model
  normalization.py         Layer normalization
  batch_normalization.py   Batch normalization
  rms_normalization.py     RMS normalization
  embeddings.py            Word embeddings
  positional_encoding.py   Positional encoding
  kv_cache.py              KV-Cache for fast inference
  grouped_query_attention.py  Grouped query attention

data/           Data pipeline
  tokenizer.py                BPE tokenizer
  vocab.py                    Character-level vocabulary
  loader.py                   Batched training data loader
  dataset.py                  GPT dataset preparation
  nlp_preprocessing.py        NLP preprocessing
  tokenizer_utils.py          Tokenization edge cases

train.py        GPT training loop
generate.py     Text generation

foundations/    Neural network primitives built from scratch
  neuron.py, backprop.py, mlp.py, activations.py, loss.py,
  training_loop.py, dead_relu_detector.py, ...
```

## Quick Start

```bash
pip install -r requirements.txt
python train.py
python generate.py
```

## Course

This project was built by completing the [NeetCode ML Course](https://neetcode.io/practice?tab=coreSkills&topic=Machine+Learning):
- Math Foundations (gradient descent, activations, loss functions)
- Neural Networks from scratch (neuron, backprop, MLP)
- PyTorch fundamentals
- NLP pipeline (embeddings, tokenization, attention)
- Transformer architecture
- GPT model + text generation

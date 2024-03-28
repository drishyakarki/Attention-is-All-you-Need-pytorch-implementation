# Attention Is All You Need - pytorch transformer implementation

This repository contains an implementation of the Transformer model architecture proposed in the paper "Attention is All You Need" by Vaswani et al. The Transformer model, introduced in this paper, has revolutionized the field of natural language processing (NLP) by introducing a novel architecture based entirely on self-attention mechanisms, eliminating the need for recurrent or convolutional layers. This repository is heavily inspired by [Umar Jamil](https://github.com/hkproj/pytorch-transformer/)'s pytorch implementation. For detailed description of the paper, you can check out his amazing [video on transformer](https://www.youtube.com/watch?v=ISNdQcPhsts).

![Transformer Architecture](images/transformer-architecture.webp)

## Introduction

The Transformer model architecture introduced in "Attention is All You Need" has become the de facto standard for various NLP tasks such as machine translation, text generation, and sentiment analysis. This implementation aims to provide a clear and concise implementation of the Transformer model in PyTorch, along with example usage and training scripts.

## Key Features

- Implementation of the core Transformer architecture including encoder, decoder, self-attention, cross-attention, residual connection, input embeddings, multi-head attention, positional encoding, and layer normalization.
- Easy-to-use interface for building and training Transformer models for various NLP tasks.
- Customizable hyperparameters such as embedding size, number of layers, number of attention heads, and dropout rate.

## Example usage

```python
from build_transformer import build_transformer

# Build Transformer model
transformer = build_transformer(
        src_vocab_size=1024,
        tgt_vocab_size=1024,
        src_seq_len=512,
        tgt_seq_len=512,
        d_model=512,
        N=6,
        h=8,   
        d_ff=2048,
        dropout=0.1,
)

# Train the model, generate predictions, etc.
```

## Getting started
To get started with using the Transformer model implemented in this repository, you can clone the repository and follow the example usage provided above. You need to install torch and numpy.
```bash
pip install torch numpy
```
# Acknowledgements

- [The "Attention is All You Need" Paper](https://arxiv.org/abs/1706.03762)
- [Umar Jamil's PyTorch Transformer](https://github.com/hkproj/pytorch-transformer/)
- [Umar Jamil's Transformer Video](https://www.youtube.com/watch?v=ISNdQcPhsts)
- [The Illustrated Transformer by Jay Alammar](http://jalammar.github.io/illustrated-transformer/)
import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    """Construct the embeddings.
    
    Args:
        d_model: size of the embedding vector.
        vocab_size: size of the vocabulary.

    Returns:
        torch.Tensor: the embedding matrix
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """
    Generate positional encoding for input sentences.

    Args:
        d_model: size of the model/embedding space.
        seq_len: maximum length of the sequence

    Returns:
        torch.Tensor: the position encoding
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape(seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape(seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        denom_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(1000.0)/d_model))
        # Apply the sin to even position
        pe[:, 0::2] = torch.sin(position * denom_term)
        # Apply the cos to odd position
        pe[:, 1::2] = torch.cos(position * denom_term)

        # Add batch dimension to pe
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        # Save the tensors in the file along with the module
        # Buffers are parameters that are not updated by gradient descent but are still part of the model state
        self.register_buffer('pe', pe) 

    def forward(self, x):
        # Add each element of x to the corresponding element of positional encoding tensor
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    """
    Applies layer normalization over a mini-batch of inputs.

    Args:
        eps: A value added for numerical stability

    Returns:
        torch.Tensor: Normalized input tensor
    """
    def __init__(self, features: int, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        # Initialize learnable parameters
        self.alpha = nn.Parameter(torch.ones(features)) # Multiplicative
        self.bias = nn.Parameter(torch.zeros(features)) # Additive

    def forward(self, x):
        mean = x.mean(dim = -1, keepdims = True)
        std = x.std(dim = -1, keepdims = True)
        return self.alpha * (x - mean) / (std - self.eps) + self.bias
    
class FeedForwardLayer(nn.Module):
    """
    Feedforward layer in a Transformer model.

    Args:
        d_model: The dimensionality of the input and output vectors (embedding size).
        d_ff: The dimensionality of the intermediate layer (hidden layer size).
        dropout: Dropout probability for regularization.

    Returns:
        torch.Tensor: Transformed output tensor.

    Shape:
        - Input: `(Batch, seq_len, d_model)`
        - Output: `(Batch, seq_len, d_model)`
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1)))
    
class MultiHeadAttention(nn.Module):
    """
     Multi-Head Attention mechanism module.

    Args:
        d_model: Dimensionality of the input and output embeddings.
        h: Number of attention heads.
        dropout: Dropout probability for regularization.

    Returns:
        torch.Tensor: Output tensor after multi-head attention computation.
    """
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        #(batch, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        # (batch, h, seq_len, seq_len)
        attention_scores = attention_scores.softmax(dim=-1)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return attention_scores @ value

    def forward(self, q, k, v, mask):
        query = self.w_q(q) #(batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) #(batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) #(batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    """
    Implements a residual connection with layer normalization and dropout.

    Args:
        dropout: Dropout probability for regularization.

    Returns:
        torch.Tensor: Output tensor after applying the residual connection.
    """
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardLayer, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
        
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    """
    Transformer encoder composed of multiple encoder blocks.

    Args:
        layers (nn.ModuleList): List of encoder blocks.
        
    Returns:
        torch.Tensor: Output tensor after encoding.
    """
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardLayer, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    """
    Transformer decoder composed of multiple decoder blocks.

    Args:
        layers (nn.ModuleList): List of decoder blocks.
        
    Returns:
        torch.Tensor: Output tensor after decoding.
    """
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    """
    Projection layer to map model output to vocabulary space.

    Args:
        d_model (int): Size of the model's output embedding.
        vocab_size (int): Size of the vocabulary.

    Returns:
        torch.Tensor: Log softmax of the projected tensor.
    """
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)
    
class Transformer(nn.Module):
    """
    Transformer model composed of encoder, decoder, and projection layer.

    Args:
        encoder (Encoder): Encoder module.
        decoder (Decoder): Decoder module.
        src_embed (InputEmbedding): Source input embedding module.
        tgt_embed (InputEmbedding): Target input embedding module.
        src_pos (PositionalEncoding): Positional encoding module for source inputs.
        tgt_pos (PositionalEncoding): Positional encoding module for target inputs.
        projection_layer (ProjectionLayer): Projection layer module.

    Returns:
        torch.Tensor: Log softmax of the projected tensor.
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
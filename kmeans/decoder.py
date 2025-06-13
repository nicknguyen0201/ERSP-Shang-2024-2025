import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding module."""
    # changed max_len from 5000 to 10000 for longer datasets
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:  # if odd, adjust cosine dimension
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
"""
This is the gausian noise function, uses to add some noise to the dataset to prevent overfiting,
the noise is calculated with
1. randn_like(x) random numbers shape to tensor x from gaussian distribution with mean =0 and stdev =1
2. Multiply by sigma, the chosen standard deviation for the noise
sigma can be arbitrarily adjusted, but because the dataset is already normalized so chat gpt recommend starting with sigma=0.1
"""
class GaussianNoise(nn.Module):
    def __init__(self, sigma ):
        super().__init__()
        self.sigma = sigma
        
    def forward(self, x):
        if self.training and self.sigma!=0:
            noise = torch.randn_like(x)*self.sigma
            return x+noise
        # don't add noise if not inside training process
        return x
        
class TransformerDecoder(nn.Module):
    """
    Transformer-based decoder to replace the MLP decoder.
    
    Instead of a simple sequential block, this module uses a transformer decoder.
    A set of learnable query embeddings defines the number of output tokens (i.e. the output sequence length).
    """
    def __init__(self, ninp, nhid, decoder_n_out, num_queries, num_layers=5, nhead=4, dropout=0.0,noise_sigma=0.0, **kwargs):
        """
        Args:
            ninp (int): Input dimension (should match model.ninp from TabPFN).
            nhid (int): Hidden dimension (should match model.nhid).
            decoder_n_out (int): Output dimension per token (e.g. centroid dimension).
            num_queries (int): Number of queries; controls the number of output tokens (variable output length).
            num_layers (int): Number of transformer decoder layers.
            nhead (int): Number of attention heads.
            dropout (float): Dropout probability.
            noise_sigma (float): intensity of the noise
        """
        super().__init__()
        # Project the memory (e.g. latent output from TabPFN) into hidden space.
        self.input_linear = nn.Linear(ninp, nhid)
        # init the gaussian noise
        self.gaussian_noise=GaussianNoise(noise_sigma)    
        # Positional encoding for the memory sequence.
        self.pos_encoder = PositionalEncoding(nhid, dropout)
        print (f"dropout is ", dropout)
        # Learnable query embeddings define the output tokens.
        self.query_embed = nn.Embedding(num_queries, nhid)
        
        # Build the transformer decoder layers.
        decoder_layer = nn.TransformerDecoderLayer(d_model=nhid, nhead=nhead, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Final projection to get the desired output dimension.
        self.fc_out = nn.Linear(nhid, decoder_n_out)
    
    def forward(self, memory, memory_mask=None, tgt_mask=None, **kwargs):
        """
        Args:
            memory (Tensor): Latent representation from the TabPFN model.
                             Expected shape: (memory_seq_len, batch_size, ninp)
            memory_mask (optional Tensor): Mask for the memory sequence.
            tgt_mask (optional Tensor): Mask for the target queries.
        Returns:
            Tensor: Output sequence of shape (num_queries, batch_size, decoder_n_out)
        """
        # Project and add positional encodings to memory.
        memory = self.input_linear(memory)             # -> (memory_seq_len, batch_size, nhid)
        #new gaussian noise
        memory =self.gaussian_noise(memory)

        memory = self.pos_encoder(memory)
        
        batch_size = memory.size(1)
        # Prepare query tokens from learnable embeddings.
        tgt = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)  # (num_queries, batch_size, nhid)
        
        # Decode using the transformer decoder.
        decoded = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        
        # Final linear projection.
        output = self.fc_out(decoded)
        return output

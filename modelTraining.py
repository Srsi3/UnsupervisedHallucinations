import torch
import torch.nn as nn

class VQVAE(nn.Module):
    def __init__(self, codebook_size=1024, code_dim=256):
        super().__init__()
        # Encoder: downsamples input to latent feature map
        self.encoder = EncoderCNN()  # define your conv encoder
        # Codebook: an embedding layer with `codebook_size` vectors of dimension `code_dim`
        self.codebook = nn.Embedding(codebook_size, code_dim)
        # Initialize codebook embeddings
        nn.init.uniform_(self.codebook.weight, -1./codebook_size, 1./codebook_size)
        # Decoder: upsamples from latent code vectors to image
        self.decoder = DecoderCNN()  # define your conv decoder

    def encode(self, x):
        z_e = self.encoder(x)  # shape: (B, D, H, W)
        # Permute to (B, H, W, D) for convenience
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        B, H, W, D = z_e.shape
        # Flatten spatial dimensions
        z_flat = z_e.view(B * H * W, D)
        # Compute L2 distance between each encoded vector and each codebook embedding
        # distances: shape (B*H*W, codebook_size)
        z_sq = (z_flat ** 2).sum(dim=1, keepdim=True)               # ||z||^2
        e_sq = (self.codebook.weight ** 2).sum(dim=1, keepdim=True) # ||e||^2 for each code e
        # We use (a - b)^2 = a^2 + b^2 - 2ab for efficiency:
        distances = z_sq + e_sq.t() - 2 * (z_flat @ self.codebook.weight.t())
        # Find nearest embedding index for each latent vector
        encoding_indices = torch.argmin(distances, dim=1)  # (B*H*W,)
        codes = encoding_indices.view(B, H, W)  # reshape back to image grid of codes
        return codes, encoding_indices

    def decode(self, codes):
        # Convert code indices back to quantized vectors
        # codes shape: (B, H, W), we first flatten to (B*H*W) and get embeddings
        B, H, W = codes.shape
        code_flat = codes.view(B * H * W)
        z_q = self.codebook(code_flat)  # (B*H*W, code_dim)
        # Reshape back to (B, D, H, W)
        z_q = z_q.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_recon = self.decoder(z_q)
        return x_recon

    def forward(self, x):
        codes, encoding_indices = self.encode(x)
        x_recon = self.decode(codes)
        return x_recon, codes


class MatFormerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_ratio=4.0, sub_ffn_ratios=[2.0, 1.0]):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        # Main FFN
        d_ffn = int(d_model * ffn_ratio)
        self.ffn_main = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, d_model)
        )
        # Sub-FFNs (with smaller ratios)
        self.ffn_subs = nn.ModuleList()
        for ratio in sub_ffn_ratios:
            d_sub = int(d_model * ratio)
            # Each sub-FFN could either be a separate module or share weights with main
            # For simplicity, we'll make them separate smaller networks:
            self.ffn_subs.append(nn.Sequential(
                nn.Linear(d_model, d_sub),
                nn.GELU(),
                nn.Linear(d_sub, d_model)
            ))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, submodel=None):
        # Self-attention block
        attn_out, _ = self.self_attn(x, x, x)
        x = x + attn_out
        x = self.norm1(x)
        # Feed-forward block
        if submodel is None:
            # use full model
            ffn_out = self.ffn_main(x)
        else:
            # use one of the sub-models (by index)
            ffn_out = self.ffn_subs[submodel](x)
        x = x + ffn_out
        x = self.norm2(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

dtype = torch.float32


class SiameseNetwork(nn.Module):
    def __init__(self, encoder, latent_dim=128, dropout=0, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(latent_dim*4, latent_dim)
        )

    def forward(self, x):
        emb = self.encoder(x)
        emb = self.projector(emb)
        return emb

import torch
import torch.nn as nn
import torch.nn.functional as F

dtype = torch.float32


class DDIPredictor(nn.Module):
    def __init__(self, encoder, latent_dim=128, num_tasks=1, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim*2, latent_dim*2), nn.ReLU(), nn.Dropout(kwargs['dropout']), nn.Linear(latent_dim*2, num_tasks)
        )

    def forward(self, mol1, mol2):
        emb1 = self.encoder(mol1)
        emb2 = self.encoder(mol2)
        emb = torch.cat((emb1, emb2), dim=-1)
        out = self.predictor(emb)

        return out

    def from_pretrained(self, model_path, device):
        pre_model = torch.load(model_path, map_location=device)
        self.encoder.load_state_dict(pre_model)

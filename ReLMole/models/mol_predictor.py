import torch
import torch.nn as nn
import torch.nn.functional as F

dtype = torch.float32


class MolPredictor(nn.Module):
    def __init__(self, encoder, latent_dim=128, num_tasks=1, **kwargs):
        super().__init__()

        self.encoder = encoder
        self.predictor = nn.Linear(latent_dim, num_tasks)

    def forward(self, mol):
        emb = self.encoder(mol)
        out = self.predictor(emb)

        return out

    def from_pretrained(self, model_path, device):
        pre_model = torch.load(model_path, map_location=device)
        self.encoder.load_state_dict(pre_model)

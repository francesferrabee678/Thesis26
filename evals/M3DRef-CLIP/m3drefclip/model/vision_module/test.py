import torch
import torch.nn as nn
from torchvision.transforms import Normalize
import lightning.pytorch as pl


class DINOv2ImageEncoder(pl.LightningModule):
    def __init__(self, dino_model, output_channel, dropout):

        super().__init__()
        self.dino_model = dino_model  
        self.dino_dim = self.dino_model.embed_dim      
        self.mlp = nn.Sequential(
            nn.Linear(self.dino_dim, output_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_channel, output_channel),
        )
    def forward(self, x):
        x_norm = Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073), 
            std=(0.26862954, 0.26130258, 0.27577711)
        )(x)
        
        features_dict = self.dino_model.forward_features(x_norm)
        patch_tokens = features_dict["x_norm_patchtokens"] 
        
        output = torch.mean(patch_tokens, dim=1) # Shape: [Batch, 1024]
        
        output = nn.functional.normalize(output, dim=1)
        
        return self.mlp(output)

# import torch
# import torch.nn as nn
# from models.common import LSTMEncoder, RegressionHead, mean_pooling
#
# class CrossModalAttentionLSTM(nn.Module):
#     def __init__(self, hidden_dim=128, num_layers=1, dropout=0.2, num_heads=4):
#         super().__init__()
#         self.text_encoder = LSTMEncoder(300, hidden_dim, num_layers, dropout)
#         self.audio_encoder = LSTMEncoder(74, hidden_dim, num_layers, dropout)
#         self.vision_encoder = LSTMEncoder(35, hidden_dim, num_layers, dropout)
#
#         self.attn_ta = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
#         self.attn_tv = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
#
#         self.regressor = RegressionHead(hidden_dim * 3, hidden_dim=128, dropout=dropout)
#
#     def forward(self, text, audio, vision):
#         ht = self.text_encoder(text)    # (B,T,H)
#         ha = self.audio_encoder(audio)
#         hv = self.vision_encoder(vision)
#
#         attn_ta, _ = self.attn_ta(query=ht, key=ha, value=ha)
#         attn_tv, _ = self.attn_tv(query=ht, key=hv, value=hv)
#
#         pt = mean_pooling(ht)
#         pta = mean_pooling(attn_ta)
#         ptv = mean_pooling(attn_tv)
#
#         fused = torch.cat([pt, pta, ptv], dim=-1)
#         return self.regressor(fused)

import torch
import torch.nn as nn
from models.common import LSTMEncoder, RegressionHead, mean_pooling, ModalityProjection


class CrossModalAttentionLSTM(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        num_layers=1,
        dropout=0.2,
        num_heads=4,
        proj_dim=64,
    ):
        super().__init__()

        # 1) modality-specific projection to a shared dimension
        self.text_proj = ModalityProjection(input_dim=300, proj_dim=proj_dim, dropout=dropout)
        self.audio_proj = ModalityProjection(input_dim=74, proj_dim=proj_dim, dropout=dropout)
        self.vision_proj = ModalityProjection(input_dim=35, proj_dim=proj_dim, dropout=dropout)

        # 2) LSTM encoders now take projected inputs
        self.text_encoder = LSTMEncoder(
            input_dim=proj_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False
        )
        self.audio_encoder = LSTMEncoder(
            input_dim=proj_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False
        )
        self.vision_encoder = LSTMEncoder(
            input_dim=proj_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False
        )

        # 3) text attends to audio and vision
        self.attn_ta = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.attn_tv = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # 4) final regression head
        self.regressor = RegressionHead(
            input_dim=hidden_dim * 3,
            hidden_dim=128,
            dropout=dropout
        )

    def forward(self, text, audio, vision):
        # Project to shared latent dimension
        text = self.text_proj(text)      # (B, T, proj_dim)
        audio = self.audio_proj(audio)   # (B, T, proj_dim)
        vision = self.vision_proj(vision)# (B, T, proj_dim)

        # Encode sequences
        ht = self.text_encoder(text)     # (B, T, H)
        ha = self.audio_encoder(audio)   # (B, T, H)
        hv = self.vision_encoder(vision) # (B, T, H)

        # Cross-modal attention: text queries audio / vision
        attn_ta, _ = self.attn_ta(query=ht, key=ha, value=ha)
        attn_tv, _ = self.attn_tv(query=ht, key=hv, value=hv)

        # Pool over time
        pt = mean_pooling(ht)       # (B, H)
        pta = mean_pooling(attn_ta) # (B, H)
        ptv = mean_pooling(attn_tv) # (B, H)

        # Concatenate and regress
        fused = torch.cat([pt, pta, ptv], dim=-1)
        out = self.regressor(fused)
        return out
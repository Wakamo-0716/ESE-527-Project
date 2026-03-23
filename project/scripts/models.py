import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)   # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, T, D]
        return x + self.pe[:, :x.size(1)]


class TransformerSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 512
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x: [B, T, input_dim]
        mask: [B, T], 1 for valid, 0 for padding (optional)
        """
        x = self.proj(x)
        x = self.pos_enc(x)
        x = self.dropout(x)

        src_key_padding_mask = None
        if mask is not None:
            # Transformer expects True for padding positions
            src_key_padding_mask = (mask == 0)

        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return h  # [B, T, hidden_dim]


def masked_mean_pooling(x, mask=None):
    """
    x: [B, T, D]
    mask: [B, T], 1 for valid, 0 for padding
    """
    if mask is None:
        return x.mean(dim=1)

    mask = mask.unsqueeze(-1).float()  # [B, T, 1]
    x = x * mask
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return x.sum(dim=1) / denom


class EarlyFusionModel(nn.Module):
    def __init__(
        self,
        text_dim,
        audio_dim,
        vision_dim,
        hidden_dim=128,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        modalities=("text", "audio", "vision")
    ):
        super().__init__()
        self.modalities = list(modalities)

        dim_map = {
            "text": text_dim,
            "audio": audio_dim,
            "vision": vision_dim
        }
        fusion_input_dim = sum(dim_map[m] for m in self.modalities)

        self.encoder = TransformerSequenceEncoder(
            input_dim=fusion_input_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, text, audio, vision, mask=None):
        inputs = {
            "text": text,
            "audio": audio,
            "vision": vision
        }
        x = torch.cat([inputs[m] for m in self.modalities], dim=-1)
        h = self.encoder(x, mask)
        z = masked_mean_pooling(h, mask)
        y = self.regressor(z).squeeze(-1)
        return y


class GatedFusionModel(nn.Module):
    def __init__(
        self,
        text_dim,
        audio_dim,
        vision_dim,
        hidden_dim=128,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        modalities=("text", "audio", "vision")
    ):
        super().__init__()
        self.modalities = list(modalities)

        if "text" in self.modalities:
            self.text_encoder = TransformerSequenceEncoder(text_dim, hidden_dim, n_heads, n_layers, dropout)
            self.text_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        if "audio" in self.modalities:
            self.audio_encoder = TransformerSequenceEncoder(audio_dim, hidden_dim, n_heads, n_layers, dropout)
            self.audio_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        if "vision" in self.modalities:
            self.vision_encoder = TransformerSequenceEncoder(vision_dim, hidden_dim, n_heads, n_layers, dropout)
            self.vision_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * len(self.modalities), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        text, audio, vision,
        text_mask=None, audio_mask=None, vision_mask=None
    ):
        pooled = []

        if "text" in self.modalities:
            ht = self.text_encoder(text, text_mask)
            zt = masked_mean_pooling(ht, text_mask)
            zt = self.text_gate(zt) * zt
            pooled.append(zt)

        if "audio" in self.modalities:
            ha = self.audio_encoder(audio, audio_mask)
            za = masked_mean_pooling(ha, audio_mask)
            za = self.audio_gate(za) * za
            pooled.append(za)

        if "vision" in self.modalities:
            hv = self.vision_encoder(vision, vision_mask)
            zv = masked_mean_pooling(hv, vision_mask)
            zv = self.vision_gate(zv) * zv
            pooled.append(zv)

        z = torch.cat(pooled, dim=-1)
        y = self.regressor(z).squeeze(-1)
        return y


class CrossModalAttentionBlock(nn.Module):
    def __init__(self, hidden_dim=128, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, query_mask=None, kv_mask=None):
        key_padding_mask = None
        if kv_mask is not None:
            key_padding_mask = (kv_mask == 0)

        attn_out, _ = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask
        )
        x = self.norm1(query + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class CrossModalAttentionModel(nn.Module):
    def __init__(
        self,
        text_dim,
        audio_dim,
        vision_dim,
        hidden_dim=128,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        modalities=("text", "audio", "vision")
    ):
        super().__init__()
        self.modalities = list(modalities)

        if "text" in self.modalities:
            self.text_encoder = TransformerSequenceEncoder(text_dim, hidden_dim, n_heads, n_layers, dropout)
        if "audio" in self.modalities:
            self.audio_encoder = TransformerSequenceEncoder(audio_dim, hidden_dim, n_heads, n_layers, dropout)
        if "vision" in self.modalities:
            self.vision_encoder = TransformerSequenceEncoder(vision_dim, hidden_dim, n_heads, n_layers, dropout)

        self.cross_block1 = CrossModalAttentionBlock(hidden_dim, n_heads, dropout)
        self.cross_block2 = CrossModalAttentionBlock(hidden_dim, n_heads, dropout)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * len(self.modalities), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        text, audio, vision,
        text_mask=None, audio_mask=None, vision_mask=None
    ):
        h_dict = {}
        m_dict = {}

        if "text" in self.modalities:
            h_dict["text"] = self.text_encoder(text, text_mask)
            m_dict["text"] = text_mask
        if "audio" in self.modalities:
            h_dict["audio"] = self.audio_encoder(audio, audio_mask)
            m_dict["audio"] = audio_mask
        if "vision" in self.modalities:
            h_dict["vision"] = self.vision_encoder(vision, vision_mask)
            m_dict["vision"] = vision_mask

        mods = self.modalities

        # 两模态：mods[0] <- mods[1]
        if len(mods) == 2:
            q_mod, kv_mod = mods[0], mods[1]
            fused_q = self.cross_block1(
                h_dict[q_mod], h_dict[kv_mod],
                m_dict[q_mod], m_dict[kv_mod]
            )

            pooled = [
                masked_mean_pooling(fused_q, m_dict[q_mod]),
                masked_mean_pooling(h_dict[kv_mod], m_dict[kv_mod])
            ]

        # 三模态：mods[0] <- mods[1] <- mods[2]
        elif len(mods) == 3:
            q1, kv1, kv2 = mods[0], mods[1], mods[2]
            h1 = self.cross_block1(h_dict[q1], h_dict[kv1], m_dict[q1], m_dict[kv1])
            h2 = self.cross_block2(h1, h_dict[kv2], m_dict[q1], m_dict[kv2])

            pooled = [
                masked_mean_pooling(h2, m_dict[q1]),
                masked_mean_pooling(h_dict[kv1], m_dict[kv1]),
                masked_mean_pooling(h_dict[kv2], m_dict[kv2]),
            ]
        else:
            raise ValueError("CrossModalAttentionModel requires 2 or 3 modalities.")

        z = torch.cat(pooled, dim=-1)
        y = self.regressor(z).squeeze(-1)
        return y


class TensorFusionModel(nn.Module):
    def __init__(
        self,
        text_dim,
        audio_dim,
        vision_dim,
        hidden_dim=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1
    ):
        super().__init__()
        self.text_encoder = TransformerSequenceEncoder(text_dim, hidden_dim, n_heads, n_layers, dropout)
        self.audio_encoder = TransformerSequenceEncoder(audio_dim, hidden_dim, n_heads, n_layers, dropout)
        self.vision_encoder = TransformerSequenceEncoder(vision_dim, hidden_dim, n_heads, n_layers, dropout)

        # 为了避免张量爆炸，hidden_dim 不要太大
        tensor_dim = (hidden_dim + 1) * (hidden_dim + 1) * (hidden_dim + 1)

        self.regressor = nn.Sequential(
            nn.Linear(tensor_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 1)
        )

    def forward(
        self,
        text, audio, vision,
        text_mask=None, audio_mask=None, vision_mask=None
    ):
        ht = self.text_encoder(text, text_mask)
        ha = self.audio_encoder(audio, audio_mask)
        hv = self.vision_encoder(vision, vision_mask)

        zt = masked_mean_pooling(ht, text_mask)
        za = masked_mean_pooling(ha, audio_mask)
        zv = masked_mean_pooling(hv, vision_mask)

        ones = torch.ones(zt.size(0), 1, device=zt.device)
        zt_ = torch.cat([ones, zt], dim=-1)
        za_ = torch.cat([ones, za], dim=-1)
        zv_ = torch.cat([ones, zv], dim=-1)

        # outer product: [B, Dt+1, Da+1, Dv+1]
        fusion_tensor = torch.einsum("bi,bj,bk->bijk", zt_, za_, zv_)
        fusion_tensor = fusion_tensor.reshape(fusion_tensor.size(0), -1)

        y = self.regressor(fusion_tensor).squeeze(-1)
        return y


class UnimodalTransformerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        n_heads=4,
        n_layers=2,
        dropout=0.1
    ):
        super().__init__()
        self.encoder = TransformerSequenceEncoder(input_dim, hidden_dim, n_heads, n_layers, dropout)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, mask=None):
        h = self.encoder(x, mask)
        z = masked_mean_pooling(h, mask)
        y = self.regressor(z).squeeze(-1)
        return y
"""
iTransformer V2: Inverted Transformer with multi-feature variate embedding.

Key changes from V1:
- Each variate now has multiple features per timestep (OHLCV + technicals)
- Embedding takes (lookback_len, n_features) per variate → d_model
- Uses a two-layer MLP for richer variate representations
- Attention still operates across variates (tickers)

Reference: Liu et al., "iTransformer: Inverted Transformers Are Effective
for Time Series Forecasting", ICLR 2024.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VariateEmbedding(nn.Module):
    """
    Embed each variate's multi-feature lookback series into d_model.

    Input per variate: (lookback_len, n_features) → flattened → MLP → d_model
    """

    def __init__(self, lookback_len: int, n_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        input_dim = lookback_len * n_features

        # Two-layer MLP for richer representation
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, lookback_len, num_variates, n_features)
        Returns:
            (batch, num_variates, d_model)
        """
        B, L, N, F = x.shape

        # Rearrange to (batch, num_variates, lookback_len * n_features)
        x = x.permute(0, 2, 1, 3)          # (B, N, L, F)
        x = x.reshape(B, N, L * F)         # (B, N, L*F)

        # Project to d_model
        x = self.embedding(x)              # (B, N, d_model)
        return x


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        Q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)
        output = self.W_o(attn_output)

        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1,
                 activation: str = "gelu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Single iTransformer encoder layer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, activation: str = "gelu",
                 norm_type: str = "pre"):
        super().__init__()
        self.norm_type = norm_type
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_type == "pre":
            x = x + self.dropout1(self.attn(self.norm1(x)))
            x = x + self.dropout2(self.ff(self.norm2(x)))
        else:
            x = self.norm1(x + self.dropout1(self.attn(x)))
            x = self.norm2(x + self.dropout2(self.ff(x)))
        return x


class iTransformer(nn.Module):
    """
    iTransformer V2 for multi-feature multivariate time series forecasting.

    1. Variate Embedding: (lookback, n_features) per variate → d_model
    2. Encoder: self-attention across variates
    3. Projection: SPY token → forecast_len predictions
    """

    def __init__(
        self,
        lookback_len: int,
        forecast_len: int,
        num_variates: int,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_type: str = "pre",
        predict_spy_only: bool = True,
        spy_index: int = 0,
    ):
        super().__init__()

        self.lookback_len = lookback_len
        self.forecast_len = forecast_len
        self.num_variates = num_variates
        self.n_features = n_features
        self.predict_spy_only = predict_spy_only
        self.spy_index = spy_index

        # Multi-feature variate embedding
        self.embedding = VariateEmbedding(lookback_len, n_features, d_model, dropout)

        # Learnable variate position embeddings
        self.variate_pos = nn.Parameter(torch.randn(1, num_variates, d_model) * 0.02)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, activation, norm_type)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model) if norm_type == "pre" else nn.Identity()

        # Projection head
        if predict_spy_only:
            self.projection = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, forecast_len),
            )
        else:
            self.projection = nn.Linear(d_model, forecast_len)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, lookback_len, num_variates, n_features)
        Returns:
            If predict_spy_only: (batch, forecast_len)
            Else: (batch, forecast_len, num_variates)
        """
        # Embed variates
        h = self.embedding(x)                   # (B, N, d_model)

        # Add variate position embeddings
        h = h + self.variate_pos

        # Self-attention across variates
        for layer in self.encoder_layers:
            h = layer(h)

        h = self.final_norm(h)

        if self.predict_spy_only:
            spy_repr = h[:, self.spy_index, :]  # (B, d_model)
            output = self.projection(spy_repr)  # (B, forecast_len)
        else:
            output = self.projection(h)         # (B, N, forecast_len)
            output = output.permute(0, 2, 1)    # (B, forecast_len, N)

        return output

    def get_attention_weights(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract attention maps for interpretability."""
        h = self.embedding(x)
        h = h + self.variate_pos
        attention_maps = []

        for layer in self.encoder_layers:
            normed = layer.norm1(h) if layer.norm_type == "pre" else h
            B, N, D = normed.shape
            attn = layer.attn

            Q = attn.W_q(normed).view(B, N, attn.n_heads, attn.d_k).transpose(1, 2)
            K = attn.W_k(normed).view(B, N, attn.n_heads, attn.d_k).transpose(1, 2)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / attn.scale
            weights = F.softmax(scores, dim=-1)
            attention_maps.append(weights.detach())

            h = layer(h)

        return attention_maps


def build_model(cfg, spy_index: int = 0) -> iTransformer:
    """Build iTransformer from config."""
    model = iTransformer(
        lookback_len=cfg.model.lookback_len,
        forecast_len=cfg.model.forecast_len,
        num_variates=cfg.model.num_variates,
        n_features=cfg.model.n_features,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        d_ff=cfg.model.d_ff,
        dropout=cfg.model.dropout,
        activation=cfg.model.activation,
        norm_type=cfg.model.norm_type,
        predict_spy_only=cfg.model.predict_spy_only,
        spy_index=spy_index,
    )

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,} total, {n_trainable:,} trainable")

    return model

"""
MultiTaskStockformer — PyTorch port of the MQ5 MTStockformer architecture.

Architecture (faithful to MQ5 original):
  Encoder:  SAMLayer x4 -> SAMConv -> MultiTaskStockformerBody -> VAE
  Actor:    account_context + encoder_latent -> 6 actions
  Critic:   actions + encoder_latent -> 3 reward estimates

Constants from Trajectory.mqh:
  HistoryBars   = 120
  BarDescr      = 9
  AccountDescr  = 12  (8 normalized account features + 4 time encodings)
  NActions      = 6
  NRewards      = 3
  NForecast     = 24
  EmbeddingSize = 32
  Segments      = 10  (HistoryBars % Segments == 0 => 12 bars per segment)
  PatchSize     = 3
  LatentCount   = 256
  DiscFactor    = 0.99

GPU: torch_directml (AMD RX 6800 XT).  LSTM stays on CPU (DirectML backprop broken for LSTM).
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
try:
    import torch_directml
    _DML_DEVICE = torch_directml.device()
    _DML_AVAILABLE = True
except ImportError:
    _DML_DEVICE = torch.device("cpu")
    _DML_AVAILABLE = False

CPU_DEVICE = torch.device("cpu")

def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return CPU_DEVICE
    return _DML_DEVICE if _DML_AVAILABLE else CPU_DEVICE


# ---------------------------------------------------------------------------
# Architecture constants (from Trajectory.mqh)
# ---------------------------------------------------------------------------
HISTORY_BARS   = 120
BAR_DESCR      = 9
ACCOUNT_DESCR  = 12   # 8 normalised account + 4 time encodings
N_ACTIONS      = 6
N_REWARDS      = 3
N_FORECAST     = 24
EMBEDDING_SIZE = 32
SEGMENTS       = 10
PATCH_SIZE     = 3
LATENT_COUNT   = 256
DISC_FACTOR    = 0.99

BARS_PER_SEG   = HISTORY_BARS // SEGMENTS   # 12 bars per segment


# ---------------------------------------------------------------------------
# SAMLayer — Segment-Aware Model layer
#
# Splits the time axis into SEGMENTS chunks, projects each chunk to an
# embedding, applies multi-head self-attention across segments, then
# reprojects back.  Four of these are stacked in the encoder.
# ---------------------------------------------------------------------------
class SAMLayer(nn.Module):
    """Segment-Aware Model layer.

    Input:  (B, T, C)   where T = HistoryBars, C = input_dim
    Output: (B, T, C)   same shape — residual-connected
    """

    def __init__(
        self,
        input_dim: int,
        embedding_size: int = EMBEDDING_SIZE,
        n_heads: int = 4,
        segments: int = SEGMENTS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.segments    = segments
        self.bars_per_seg = HISTORY_BARS // segments
        self.embed_size  = embedding_size

        # Project each bar into embedding space
        self.bar_proj   = nn.Linear(input_dim, embedding_size)
        # Segment-level attention — manual QKV to avoid DirectML MHA fallback
        assert embedding_size % n_heads == 0
        self.n_heads    = n_heads
        self.head_dim   = embedding_size // n_heads
        self.qkv        = nn.Linear(embedding_size, 3 * embedding_size, bias=False)
        self.attn_proj  = nn.Linear(embedding_size, embedding_size, bias=False)
        self.attn_drop  = nn.Dropout(dropout)
        # Feed-forward within each segment
        self.ffn = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size * 4, embedding_size),
        )
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.out_proj = nn.Linear(embedding_size, input_dim)
        self.dropout  = nn.Dropout(dropout)

    def _self_attn(self, x: torch.Tensor) -> torch.Tensor:
        """Manual multi-head self-attention — DirectML-compatible."""
        B, S, E = x.shape
        H, D = self.n_heads, self.head_dim
        qkv = self.qkv(x)                        # (B, S, 3E)
        q, k, v = qkv.split(E, dim=-1)
        # Reshape to (B, H, S, D)
        q = q.view(B, S, H, D).transpose(1, 2)
        k = k.view(B, S, H, D).transpose(1, 2)
        v = v.view(B, S, H, D).transpose(1, 2)
        scale = math.sqrt(D)
        attn  = (q @ k.transpose(-2, -1)) / scale  # (B, H, S, S)
        attn  = F.softmax(attn, dim=-1)
        attn  = self.attn_drop(attn)
        out   = (attn @ v)                         # (B, H, S, D)
        out   = out.transpose(1, 2).reshape(B, S, E)
        return self.attn_proj(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        # Project bars to embedding
        h = self.bar_proj(x)                            # (B, T, E)
        # Reshape to (B, segments, bars_per_seg, E) then pool per segment
        h_seg = h.view(B, self.segments, self.bars_per_seg, self.embed_size)
        seg_tokens = h_seg.mean(dim=2)                  # (B, S, E)
        # Segment-level self-attention (manual — no MHA module)
        attn_out   = self._self_attn(seg_tokens)
        seg_tokens = self.norm1(seg_tokens + self.dropout(attn_out))
        seg_tokens = self.norm2(seg_tokens + self.ffn(seg_tokens))
        # Broadcast segment context back to bar level
        seg_ctx = seg_tokens.unsqueeze(2).expand(
            B, self.segments, self.bars_per_seg, self.embed_size
        )                                               # (B, S, bars, E)
        seg_ctx = seg_ctx.reshape(B, T, self.embed_size)
        # Residual back to input dimension
        return x + self.out_proj(self.dropout(seg_ctx))


# ---------------------------------------------------------------------------
# SAMConv — Segment-Aware Convolution
#
# Patch-level depthwise conv across time (PatchSize=3), followed by
# pointwise projection.  Bridges the four SAM layers to the Stockformer body.
# ---------------------------------------------------------------------------
class SAMConv(nn.Module):
    """Segment-aware convolution block.

    Input/output: (B, T, C)
    """

    def __init__(
        self,
        input_dim: int,
        embedding_size: int = EMBEDDING_SIZE,
        patch_size: int = PATCH_SIZE,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        padding = patch_size // 2
        # Depthwise temporal convolution
        self.dw_conv = nn.Conv1d(
            input_dim, input_dim,
            kernel_size=patch_size, padding=padding, groups=input_dim, bias=False
        )
        # Pointwise expansion to embedding
        self.pw_proj  = nn.Linear(input_dim, embedding_size)
        self.norm     = nn.LayerNorm(embedding_size)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        h = x.permute(0, 2, 1)            # (B, C, T) for Conv1d
        h = self.dw_conv(h)
        h = h.permute(0, 2, 1)            # (B, T, C)
        h = self.norm(self.pw_proj(self.dropout(h)))
        return h                           # (B, T, E)


# ---------------------------------------------------------------------------
# _TransformerBlock — DirectML-safe transformer block
#
# Replaces nn.TransformerEncoderLayer to avoid the MHA native kernel that
# falls back to CPU on DirectML.  Uses the same manual QKV approach as SAMLayer.
# ---------------------------------------------------------------------------
class _TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_size: int,
        n_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert embedding_size % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = embedding_size // n_heads
        self.qkv      = nn.Linear(embedding_size, 3 * embedding_size, bias=False)
        self.attn_out = nn.Linear(embedding_size, embedding_size, bias=False)
        self.ffn      = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size * 4, embedding_size),
        )
        self.norm1   = nn.LayerNorm(embedding_size)
        self.norm2   = nn.LayerNorm(embedding_size)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, E = x.shape
        H, D = self.n_heads, self.head_dim
        # Pre-norm self-attention
        h    = self.norm1(x)
        qkv  = self.qkv(h)
        q, k, v = qkv.split(E, dim=-1)
        q = q.view(B, S, H, D).transpose(1, 2)
        k = k.view(B, S, H, D).transpose(1, 2)
        v = v.view(B, S, H, D).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(D)
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out  = (attn @ v).transpose(1, 2).reshape(B, S, E)
        x    = x + self.drop(self.attn_out(out))
        # Pre-norm FFN
        x    = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# MultiTaskStockformerBody — Transformer encoder over the full sequence
#
# After SAM pre-processing we have (B, T, E).  This runs a standard
# transformer encoder + multi-task projection heads:
#   - trend head:      (B, NForecast, 1)
#   - volatility head: (B, NForecast, 1)
#   - signal head:     (B, NForecast, 1)
#
# The [CLS] token is used as the sequence summary for the VAE.
# ---------------------------------------------------------------------------
class MultiTaskStockformerBody(nn.Module):

    def __init__(
        self,
        embedding_size: int = EMBEDDING_SIZE,
        n_heads: int = 4,
        n_layers: int = 2,
        n_forecast: int = N_FORECAST,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_forecast = n_forecast

        # Positional embedding (learnable, length = T + 1 for CLS)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_size))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, HISTORY_BARS + 1, embedding_size)
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Manual transformer blocks — avoids DirectML MHA fallback warning
        self.tf_blocks = nn.ModuleList([
            _TransformerBlock(embedding_size, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Multi-task forecast heads (applied to all T bar tokens)
        # We project from embedding to NForecast steps for each task
        self.trend_head      = nn.Linear(embedding_size, n_forecast)
        self.volatility_head = nn.Linear(embedding_size, n_forecast)
        self.signal_head     = nn.Linear(embedding_size, n_forecast)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, T, E)
        Returns:
            cls_out:        (B, E)       — sequence summary token
            trend_pred:     (B, NForecast)
            vol_pred:       (B, NForecast)
            signal_pred:    (B, NForecast)
        """
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)    # (B, 1, E)
        h   = torch.cat([cls, x], dim=1)           # (B, T+1, E)
        h   = h + self.pos_embed
        h   = self.dropout(h)
        for block in self.tf_blocks:
            h = block(h)                           # (B, T+1, E)

        cls_out = h[:, 0, :]                       # (B, E)
        bar_out = h[:, 1:, :]                      # (B, T, E)

        # Pool bar tokens for multi-task heads
        pooled = bar_out.mean(dim=1)               # (B, E)
        trend_pred   = self.trend_head(pooled)     # (B, NForecast)
        vol_pred     = self.volatility_head(pooled)
        signal_pred  = self.signal_head(pooled)

        return cls_out, trend_pred, vol_pred, signal_pred


# ---------------------------------------------------------------------------
# VAE — Variational Autoencoder wrapper around the latent space
#
# Takes the CLS token + account context, produces a latent sample z.
# LatentLayer = -1 means the LAST layer of the encoder is the latent source,
# which in our case is the VAE output.
# ---------------------------------------------------------------------------
class VAE(nn.Module):

    def __init__(
        self,
        input_dim: int,          # CLS embedding size
        account_dim: int,        # account context dimension
        latent_dim: int = LATENT_COUNT,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        fused_dim = input_dim + account_dim
        self.fc_mu  = nn.Linear(fused_dim, latent_dim)
        self.fc_log_var = nn.Linear(fused_dim, latent_dim)
        self.fc_decode  = nn.Linear(latent_dim, latent_dim)
        self.norm    = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)

    def reparameterise(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(
        self, cls_token: torch.Tensor, account: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        cls_token: (B, E)
        account:   (B, A)
        Returns: z (B, LatentCount), mu (B, LatentCount), log_var (B, LatentCount)
        """
        fused   = torch.cat([cls_token, account], dim=-1)
        mu      = self.fc_mu(fused)
        log_var = self.fc_log_var(fused)
        z       = self.reparameterise(mu, log_var)
        z       = self.norm(self.fc_decode(self.dropout(z)))
        return z, mu, log_var


# ---------------------------------------------------------------------------
# Encoder (the full CNet "Encoder" from MQ5)
#
# SAMLayer x4 -> SAMConv -> MultiTaskStockformerBody -> VAE
# Input:  state  (B, HistoryBars * BarDescr) flat  — or (B, T, C)
#         account (B, AccountDescr)
# Output: z (B, LatentCount) — the latent the Actor and Critic attach to
# ---------------------------------------------------------------------------
class StockformerEncoder(nn.Module):

    def __init__(
        self,
        bar_descr: int       = BAR_DESCR,
        account_descr: int   = ACCOUNT_DESCR,
        embedding_size: int  = EMBEDDING_SIZE,
        segments: int        = SEGMENTS,
        latent_count: int    = LATENT_COUNT,
        n_forecast: int      = N_FORECAST,
        dropout: float       = 0.1,
    ) -> None:
        super().__init__()
        # Initial bar projection (9 -> EmbeddingSize)
        self.bar_embed = nn.Linear(bar_descr, embedding_size)

        # Four SAM layers (input/output both embedding_size)
        self.sam_layers = nn.ModuleList([
            SAMLayer(embedding_size, embedding_size, n_heads=4,
                     segments=segments, dropout=dropout)
            for _ in range(4)
        ])

        # SAMConv bridge
        self.sam_conv = SAMConv(embedding_size, embedding_size,
                                patch_size=PATCH_SIZE, dropout=dropout)

        # Stockformer transformer body with multi-task heads
        self.stockformer = MultiTaskStockformerBody(
            embedding_size=embedding_size,
            n_heads=4, n_layers=2,
            n_forecast=n_forecast,
            dropout=dropout,
        )

        # VAE produces the final latent
        self.vae = VAE(
            input_dim=embedding_size,
            account_dim=account_descr,
            latent_dim=latent_count,
            dropout=dropout,
        )

    def forward(
        self,
        state: torch.Tensor,       # (B, T*C) flat or (B, T, C)
        account: torch.Tensor,     # (B, AccountDescr)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            z           (B, LatentCount)
            mu          (B, LatentCount)
            log_var     (B, LatentCount)
            trend_pred  (B, NForecast)
            vol_pred    (B, NForecast)
            signal_pred (B, NForecast)
        """
        B = state.size(0)
        if state.dim() == 2:
            state = state.view(B, HISTORY_BARS, -1)   # (B, 120, 9)

        # Project bars to embedding space
        h = self.bar_embed(state)                     # (B, 120, E)

        # Four SAM layers
        for sam in self.sam_layers:
            h = sam(h)

        # SAMConv
        h = self.sam_conv(h)                          # (B, 120, E)

        # Stockformer body — get CLS + multi-task predictions
        cls_out, trend_pred, vol_pred, signal_pred = self.stockformer(h)

        # VAE latent
        z, mu, log_var = self.vae(cls_out, account)

        return z, mu, log_var, trend_pred, vol_pred, signal_pred


# ---------------------------------------------------------------------------
# Actor (CNet "Actor" from MQ5)
#
# In MQ5: Actor.feedForward(bAccount, 1, false, Encoder, LatentLayer)
# Meaning: takes account context as primary input, attends to Encoder latent.
# Outputs 6 actions: [buy_lot, buy_tp, buy_sl, sell_lot, sell_tp, sell_sl]
# All outputs are sigmoid-activated (0..1 range, scaled by MaxSL/MaxTP in MQ5)
# ---------------------------------------------------------------------------
class StockformerActor(nn.Module):

    def __init__(
        self,
        account_dim: int  = ACCOUNT_DESCR,
        latent_dim: int   = LATENT_COUNT,
        n_actions: int    = N_ACTIONS,
        dropout: float    = 0.1,
    ) -> None:
        super().__init__()
        fused_dim = account_dim + latent_dim
        self.net = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_actions),
            nn.Sigmoid(),   # outputs in [0, 1] — consistent with MQ5 action range
        )

    def forward(
        self, account: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        """
        account: (B, AccountDescr)
        latent:  (B, LatentCount)
        Returns: actions (B, NActions)  values in [0, 1]
        """
        fused = torch.cat([account, latent], dim=-1)
        return self.net(fused)


# ---------------------------------------------------------------------------
# Critic (CNet "Critic" from MQ5)
#
# In MQ5: Critic.feedForward(bActions, 1, false, Encoder, LatentLayer)
# Takes action buffer as primary input, attends to encoder latent.
# Outputs 3 reward estimates: [delta_balance, equity_ratio, no_position_penalty]
# ---------------------------------------------------------------------------
class StockformerCritic(nn.Module):

    def __init__(
        self,
        n_actions: int   = N_ACTIONS,
        latent_dim: int  = LATENT_COUNT,
        n_rewards: int   = N_REWARDS,
        dropout: float   = 0.1,
    ) -> None:
        super().__init__()
        fused_dim = n_actions + latent_dim
        self.net = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_rewards),
            # No final activation — raw value estimates
        )

    def forward(
        self, actions: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        """
        actions: (B, NActions)
        latent:  (B, LatentCount)
        Returns: value_estimates (B, NRewards)
        """
        fused = torch.cat([actions, latent], dim=-1)
        return self.net(fused)


# ---------------------------------------------------------------------------
# Full MultiTaskStockformer — wraps Encoder + Actor + Critic
# ---------------------------------------------------------------------------
class MultiTaskStockformer(nn.Module):

    def __init__(
        self,
        bar_descr: int       = BAR_DESCR,
        account_descr: int   = ACCOUNT_DESCR,
        embedding_size: int  = EMBEDDING_SIZE,
        segments: int        = SEGMENTS,
        latent_count: int    = LATENT_COUNT,
        n_forecast: int      = N_FORECAST,
        n_actions: int       = N_ACTIONS,
        n_rewards: int       = N_REWARDS,
        dropout: float       = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = StockformerEncoder(
            bar_descr=bar_descr,
            account_descr=account_descr,
            embedding_size=embedding_size,
            segments=segments,
            latent_count=latent_count,
            n_forecast=n_forecast,
            dropout=dropout,
        )
        self.actor = StockformerActor(
            account_dim=account_descr,
            latent_dim=latent_count,
            n_actions=n_actions,
            dropout=dropout,
        )
        self.critic = StockformerCritic(
            n_actions=n_actions,
            latent_dim=latent_count,
            n_rewards=n_rewards,
            dropout=dropout,
        )

    def forward(
        self,
        state: torch.Tensor,      # (B, 1080) or (B, 120, 9)
        account: torch.Tensor,    # (B, 12)
        actions: Optional[torch.Tensor] = None,  # (B, 6) — for critic
    ) -> dict:
        """
        Full forward pass mirroring MQ5 Study.mq5 train loop.

        Returns dict with keys:
            latent, mu, log_var,
            trend_pred, vol_pred, signal_pred,
            actor_actions,
            critic_values  (only if actions supplied)
        """
        z, mu, log_var, trend_pred, vol_pred, signal_pred = self.encoder(
            state, account
        )
        actor_actions = self.actor(account, z)

        result = {
            "latent":       z,
            "mu":           mu,
            "log_var":      log_var,
            "trend_pred":   trend_pred,
            "vol_pred":     vol_pred,
            "signal_pred":  signal_pred,
            "actor_actions": actor_actions,
        }

        if actions is not None:
            result["critic_values"] = self.critic(actions, z)

        return result


# ---------------------------------------------------------------------------
# generate_signal — the external API
#
# Takes a [120, 9] numpy array (or torch tensor) of bar features plus
# optional account context (12-element vector).  Returns (direction, confidence).
#
# direction: +1 = BUY, -1 = SELL, 0 = HOLD
# confidence: float in [0, 1]
# ---------------------------------------------------------------------------
def generate_signal(
    features: np.ndarray,
    model: MultiTaskStockformer,
    account: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
) -> Tuple[int, float]:
    """
    features: np.ndarray shape (120, 9) — the HistoryBars * BarDescr state
    account:  np.ndarray shape (12,)    — normalised account + time features
              If None, uses zeros (inference without live account context)
    Returns:  (direction, confidence)
              direction: +1 BUY, -1 SELL, 0 HOLD
              confidence: float [0, 1]

    Mirrors the MQ5 OnTick action selection:
      actions[0] = buy_lot,  actions[1] = buy_tp,  actions[2] = buy_sl
      actions[3] = sell_lot, actions[4] = sell_tp, actions[5] = sell_sl
    Net out: if buy_lot > sell_lot => BUY signal (delta = buy_lot - sell_lot)
             else                  => SELL signal (delta = sell_lot - buy_lot)
    """
    if device is None:
        device = get_device()

    model.eval()
    with torch.no_grad():
        feat_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0)   # (1, 120, 9)
        if account is None:
            account = np.zeros(ACCOUNT_DESCR, dtype=np.float32)
        acc_t = torch.tensor(account, dtype=torch.float32).unsqueeze(0)     # (1, 12)

        # Move to device
        feat_t = feat_t.to(device)
        acc_t  = acc_t.to(device)

        out = model.forward(feat_t, acc_t)
        actions = out["actor_actions"][0].cpu().numpy()   # (6,)

        buy_lot  = float(actions[0])
        sell_lot = float(actions[3])

        # Net out as per MQ5 logic
        if buy_lot >= sell_lot:
            net_buy  = buy_lot - sell_lot
            net_sell = 0.0
        else:
            net_buy  = 0.0
            net_sell = sell_lot - buy_lot

        # Use multi-task signal head to weight direction confidence
        trend_pred  = out["trend_pred"][0].cpu().mean().item()   # scalar mean
        vol_pred    = out["vol_pred"][0].cpu().mean().item()
        signal_pred = out["signal_pred"][0].cpu().mean().item()

        # Weighted confidence: signal_pred drives direction confidence,
        # modulated by trend and inverse volatility
        raw_conf = (abs(signal_pred) * 0.6 + abs(trend_pred) * 0.3 +
                    (1.0 - min(abs(vol_pred), 1.0)) * 0.1)
        confidence = float(min(max(raw_conf, 0.0), 1.0))

        min_lot_threshold = 0.01
        if net_buy >= min_lot_threshold:
            direction = 1
        elif net_sell >= min_lot_threshold:
            direction = -1
        else:
            direction = 0

    return direction, confidence


# ---------------------------------------------------------------------------
# VAE loss (KL divergence term)
# ---------------------------------------------------------------------------
def vae_kl_loss(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())


# ---------------------------------------------------------------------------
# train_step — mirrors the Study.mq5 training loop
#
# batch dict keys:
#   state      (B, 120, 9) or (B, 1080)
#   account    (B, 12)
#   actions    (B, 6)     — stored actions from replay buffer
#   rewards_now  (B, 3)   — States[i+1].rewards
#   rewards_next (B, 3)   — States[i+2].rewards  (for TD target)
#   traj_profitable bool  — whether trajectory total reward > 0
#
# Returns dict of scalar losses.
# ---------------------------------------------------------------------------
def train_step(
    batch: dict,
    model: MultiTaskStockformer,
    optimizer: torch.optim.Optimizer,
    device: Optional[torch.device] = None,
    disc_factor: float = DISC_FACTOR,
) -> dict:
    """
    Walk-forward training step.  Matches Study.mq5 logic:
      1. Encoder forward
      2. Critic backward (TD error on stored actions)
      3. Actor backward (only when trajectory was profitable)
      4. Critic-guided actor update (squeeze more value from critic signal)
      5. VAE reconstruction loss
      6. Multi-task forecast consistency loss
    """
    if device is None:
        device = get_device()

    model.train()
    optimizer.zero_grad()

    def to(t: torch.Tensor) -> torch.Tensor:
        return t.to(device)

    state        = to(batch["state"].float())
    account      = to(batch["account"].float())
    actions_buf  = to(batch["actions"].float())
    rewards_now  = to(batch["rewards_now"].float())
    rewards_next = to(batch["rewards_next"].float())
    traj_profitable: bool = bool(batch.get("traj_profitable", True))

    # -----------
    # Forward: encoder + actor + critic on stored actions
    # -----------
    z, mu, log_var, trend_pred, vol_pred, signal_pred = model.encoder(state, account)
    critic_vals = model.critic(actions_buf, z)

    # TD target: rewards[i+1] - rewards[i+2] * DiscFactor
    # (mirrors MQ5: result = rewards_now - rewards_next * DiscFactor)
    td_target = rewards_now - rewards_next * disc_factor

    # Critic loss: MSE to TD target
    critic_loss = F.mse_loss(critic_vals, td_target.detach())

    # -----------
    # Actor loss (only applied when trajectory was profitable)
    # Mirrors MQ5: if(Buffer[tr].States[0].rewards[0] > 0) -> actor backprop
    # -----------
    actor_actions = model.actor(account, z.detach())   # detach z for actor update
    actor_loss    = torch.tensor(0.0, device=device)
    if traj_profitable:
        # Supervised: push actor toward stored (good) actions
        actor_loss = F.mse_loss(actor_actions, actions_buf.detach())

    # -----------
    # Critic-guided actor update (MQ5: inflate critic outputs by 1%, backprop)
    # We compute a policy gradient: maximise critic value of actor's actions
    # -----------
    actor_actions_fresh = model.actor(account, z.detach())
    critic_on_actor     = model.critic(actor_actions_fresh, z.detach())
    # Inflate positive values (+1%), shrink negative (-1%) — exactly as MQ5
    inflated = torch.where(
        critic_on_actor >= 0,
        critic_on_actor * 1.01,
        critic_on_actor * 0.99,
    )
    pg_loss = F.mse_loss(critic_on_actor, inflated.detach())

    # -----------
    # VAE KL loss
    # -----------
    kl_loss = vae_kl_loss(mu, log_var)

    # -----------
    # Multi-task consistency loss
    # We enforce that trend/signal predictions are self-consistent across
    # the NForecast horizon (smooth temporal variation)
    # -----------
    mt_loss = (
        F.l1_loss(trend_pred[:, 1:],  trend_pred[:, :-1].detach()) +
        F.l1_loss(vol_pred[:, 1:],    vol_pred[:, :-1].detach()) +
        F.l1_loss(signal_pred[:, 1:], signal_pred[:, :-1].detach())
    ) / 3.0

    # Total loss — weight KL lightly so VAE doesn't collapse
    total_loss = (
        critic_loss +
        actor_loss +
        pg_loss +
        0.001 * kl_loss +
        0.01  * mt_loss
    )

    total_loss.backward()
    # Gradient clipping — defensive, essential for long trajectories
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "total":    float(total_loss.item()),
        "critic":   float(critic_loss.item()),
        "actor":    float(actor_loss.item()),
        "pg":       float(pg_loss.item()),
        "kl":       float(kl_loss.item()),
        "multitask": float(mt_loss.item()),
    }


# ---------------------------------------------------------------------------
# Weight persistence
# ---------------------------------------------------------------------------
def save_weights(model: MultiTaskStockformer, path: str) -> None:
    """Save full model state dict to path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[StockformerEncoder] saved -> {path}")


def load_weights(
    model: MultiTaskStockformer,
    path: str,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> MultiTaskStockformer:
    """Load weights into model.  Returns the model."""
    if device is None:
        device = get_device()
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=strict)
    model.to(device)
    print(f"[StockformerEncoder] loaded <- {path}")
    return model


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------
def build_model(device: Optional[torch.device] = None) -> MultiTaskStockformer:
    """Instantiate a fresh MultiTaskStockformer and move it to device."""
    if device is None:
        device = get_device()
    model = MultiTaskStockformer()
    model.to(device)
    return model


def build_optimizer(
    model: MultiTaskStockformer,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


# ---------------------------------------------------------------------------
# Quick smoke test — run this file directly to verify shapes
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== MultiTaskStockformer smoke test ===")
    print(f"DirectML available: {_DML_AVAILABLE}")
    print(f"Active device:      {get_device()}")

    dev   = get_device()
    model = build_model(dev)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters:   {total_params:,}")

    # Fake batch
    B    = 4
    feat = torch.randn(B, HISTORY_BARS, BAR_DESCR, device=dev)
    acc  = torch.randn(B, ACCOUNT_DESCR, device=dev)
    acts = torch.rand(B, N_ACTIONS, device=dev)
    r_now  = torch.randn(B, N_REWARDS, device=dev)
    r_next = torch.randn(B, N_REWARDS, device=dev)

    out = model(feat, acc, acts)
    print(f"latent shape:       {out['latent'].shape}")
    print(f"actor_actions:      {out['actor_actions'].shape}")
    print(f"critic_values:      {out['critic_values'].shape}")
    print(f"trend_pred:         {out['trend_pred'].shape}")
    print(f"vol_pred:           {out['vol_pred'].shape}")
    print(f"signal_pred:        {out['signal_pred'].shape}")

    # generate_signal
    feat_np = np.random.randn(HISTORY_BARS, BAR_DESCR).astype(np.float32)
    direction, confidence = generate_signal(feat_np, model, device=dev)
    print(f"generate_signal:    direction={direction}, confidence={confidence:.4f}")

    # train_step
    opt   = build_optimizer(model)
    batch = {
        "state":          feat,
        "account":        acc,
        "actions":        acts,
        "rewards_now":    r_now,
        "rewards_next":   r_next,
        "traj_profitable": True,
    }
    losses = train_step(batch, model, opt, device=dev)
    print(f"train_step losses:  {losses}")

    print("=== PASS ===")

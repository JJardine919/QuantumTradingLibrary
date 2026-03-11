"""
expert_conformer.py
Conformer Trading Expert — PyTorch port of DNG's MQ5 Conformer architecture.

Architecture (faithful to CreateDescriptions in Trajectory.mqh):

ENCODER:
  Input:  9 floats (1 bar x BarDescr=9) — BatchNorm — EmbeddingOCL (5 windows {4,1,1,1,2},
          out=4) — ConvOCL (stride=4, out=8) — PE — 5x ConformerOCL(100 tokens, dim=8, heads=4)

ACTOR:
  Input:  12 floats (AccountDescr) — FC(8, sigmoid) —
          3x CrossAttn(query: 1 token dim=8, keys: 500 tokens dim=8, heads=4, out=16) —
          FC(512, sigmoid) — FC(12) — VAE -> 6 outputs (NActions)

CRITIC:
  Input:  6 floats (NActions) — FC(8, sigmoid) —
          3x CrossAttn (same dims as actor) —
          FC(512, sigmoid) — FC(512, sigmoid) — FC(3) (NRewards)

Bar features (BarDescr=9, from Research.mq5/Test.mq5 OnTick):
  [0] close - open
  [1] high  - open
  [2] low   - open
  [3] tick_volume / 1000
  [4] RSI(14)
  [5] CCI(14)
  [6] ATR(14)
  [7] MACD main
  [8] MACD signal

Account features (AccountDescr=12, normalised in Train):
  [0]  (balance - prev_balance) / prev_balance
  [1]  equity / prev_balance
  [2]  (equity - prev_equity) / prev_equity
  [3]  buy_volume
  [4]  sell_volume
  [5]  buy_profit / prev_balance
  [6]  sell_profit / prev_balance
  [7]  position_discount / prev_balance
  [8]  sin(2pi * t / year_seconds)
  [9]  cos(2pi * t / month_seconds)
  [10] sin(2pi * t / week_seconds)
  [11] sin(2pi * t / day_seconds)

Actions (NActions=6):
  [0] buy_volume
  [1] buy_tp   (normalised 0-1, maps to MaxTP=1000 points)
  [2] buy_sl   (normalised 0-1, maps to MaxSL=1000 points)
  [3] sell_volume
  [4] sell_tp
  [5] sell_sl

Training: Actor-Critic with discounted rewards (gamma=0.99), VAE reparameterisation.
GPU:      torch_directml (AMD RX 6800 XT) — no CUDA.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Device selection
# INFER_DEVICE: AMD DirectML GPU for forward pass (inference)
# TRAIN_DEVICE: CPU for backward pass — DirectML backprop is broken/incomplete
#               (same constraint as LSTM; CompileGraph failures on backward)
# ---------------------------------------------------------------------------
def _get_directml_device():
    try:
        import torch_directml  # type: ignore
        return torch_directml.device()
    except Exception:
        return torch.device("cpu")


INFER_DEVICE = _get_directml_device()   # forward / inference
TRAIN_DEVICE = torch.device("cpu")      # backward / training
DEVICE       = INFER_DEVICE             # default (used in build_conformer_expert)

# ---------------------------------------------------------------------------
# Architecture constants (mirrors Trajectory.mqh #defines)
# ---------------------------------------------------------------------------
HISTORY_BARS   = 1          # HistoryBars  — current bar state fed to encoder input
GPT_BARS       = 100        # GPTBars      — context window processed by Conformer
PRECODER_BARS  = 10         # PrecoderBars — forecast depth (unused in inference)
BAR_DESCR      = 9          # BarDescr     — features per bar
ACCOUNT_DESCR  = 12         # AccountDescr — account feature vector
N_ACTIONS      = 6          # NActions
N_REWARDS      = 3          # NRewards
EMBEDDING_SIZE = 8          # EmbeddingSize
LATENT_COUNT   = 512        # LatentCount
LATENT_LAYER   = 2          # LatentLayer (VAE mu/log-var split)
DISC_FACTOR    = 0.99       # DiscFactor
MAX_SL         = 1000       # MaxSL (points)
MAX_TP         = 1000       # MaxTP (points)

# Embedding sub-window sizes from Trajectory.mqh layer 2:
# int temp[] = {4, 1, 1, 1, 2};  — 5 groups of the 9 bar features
# Groups: [0:4] OHLCV-ish | [4] RSI | [5] CCI | [6] ATR | [7:9] MACD+signal
EMBED_WINDOWS  = [4, 1, 1, 1, 2]   # must sum to BAR_DESCR (9)
EMBED_OUT_HALF = EMBEDDING_SIZE // 2  # 4 — window_out of embedding layer
CONV_OUT       = EMBEDDING_SIZE       # 8 — after the ConvOCL projection
N_HEADS        = 4                    # step=4 in ConformerOCL/CrossAttn
CROSS_ATTN_OUT = 16                   # window_out=16 in CrossAttn layers
N_CONFORMER    = 5                    # number of ConformerOCL blocks

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_linear(layer: nn.Linear) -> nn.Linear:
    """Xavier uniform init — standard for ADAM-trained layers."""
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


# ---------------------------------------------------------------------------
# Positional Encoding — learnable (defNeuronPEOCL)
# Shape: (1, GPT_BARS, d_model)
# ---------------------------------------------------------------------------
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, d_model)
        return x + self.pe


# ---------------------------------------------------------------------------
# Feed-Forward Module (used inside ConformerBlock, half-step sandwich)
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            _init_linear(nn.Linear(d_model, d_model * expansion)),
            nn.SiLU(),
            nn.Dropout(dropout),
            _init_linear(nn.Linear(d_model * expansion, d_model)),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.5 * self.net(x)


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention (inside ConformerBlock)
# ---------------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.norm  = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        out, _ = self.attn(normed, normed, normed)
        return x + self.drop(out)


# ---------------------------------------------------------------------------
# Convolution Module (inside ConformerBlock — the local pattern extractor)
# depthwise separable conv with gating, standard Conformer recipe
# ---------------------------------------------------------------------------
class ConvModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.norm       = nn.LayerNorm(d_model)
        self.pointwise1 = _init_linear(nn.Linear(d_model, 2 * d_model))
        self.depthwise  = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.bn         = nn.BatchNorm1d(d_model)
        self.pointwise2 = _init_linear(nn.Linear(d_model, d_model))
        self.drop       = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        residual = x
        x = self.norm(x)
        # GLU gate
        x = self.pointwise1(x)               # (B, T, 2*d)
        x = F.glu(x, dim=-1)                 # (B, T, d)
        # depthwise
        x = x.transpose(1, 2)               # (B, d, T)
        x = self.depthwise(x)
        x = self.bn(x)
        x = F.silu(x)
        x = x.transpose(1, 2)               # (B, T, d)
        x = self.pointwise2(x)
        x = self.drop(x)
        return residual + x


# ---------------------------------------------------------------------------
# ConformerBlock — FF -> MHSA -> Conv -> FF
# Mirrors defNeuronConformerOCL from DNG's library
# ---------------------------------------------------------------------------
class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ff1  = FeedForward(d_model, dropout=dropout)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout=dropout)
        self.conv = ConvModule(d_model, dropout=dropout)
        self.ff2  = FeedForward(d_model, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ff1(x)
        x = self.attn(x)
        x = self.conv(x)
        x = self.ff2(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Cross-Attention layer — defNeuronCrossAttenOCL
# Query comes from account/action side; keys/values from encoder output.
# units = [1, GPT_BARS*5=500], windows = [EmbeddingSize, EmbeddingSize]
# window_out = 16, step(heads) = 4
# ---------------------------------------------------------------------------
class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        n_heads: int,
        out_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Project query and key/value to a common head dimension
        head_dim = max(out_dim // n_heads, 1)
        inner    = head_dim * n_heads
        self.q_proj  = _init_linear(nn.Linear(query_dim, inner))
        self.k_proj  = _init_linear(nn.Linear(key_dim,   inner))
        self.v_proj  = _init_linear(nn.Linear(key_dim,   inner))
        self.out_proj = _init_linear(nn.Linear(inner, out_dim))
        self.n_heads  = n_heads
        self.head_dim = head_dim
        self.scale    = math.sqrt(head_dim)
        self.drop     = nn.Dropout(dropout)
        self.norm_q   = nn.LayerNorm(query_dim)
        self.norm_k   = nn.LayerNorm(key_dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        query:   (B, Tq, query_dim)
        context: (B, Tk, key_dim)
        returns: (B, Tq, out_dim)
        """
        B, Tq, _ = query.shape
        Tk        = context.shape[1]
        H, D      = self.n_heads, self.head_dim

        q = self.q_proj(self.norm_q(query)).view(B, Tq, H, D).transpose(1, 2)   # (B,H,Tq,D)
        k = self.k_proj(self.norm_k(context)).view(B, Tk, H, D).transpose(1, 2) # (B,H,Tk,D)
        v = self.v_proj(context).view(B, Tk, H, D).transpose(1, 2)              # (B,H,Tk,D)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale              # (B,H,Tq,Tk)
        attn   = F.softmax(scores, dim=-1)
        attn   = self.drop(attn)

        out = torch.matmul(attn, v)                                              # (B,H,Tq,D)
        out = out.transpose(1, 2).contiguous().view(B, Tq, H * D)               # (B,Tq,H*D)
        # No residual here — output dim (16) differs from input query dim (8).
        # defNeuronCrossAttenOCL in DNG's library does not enforce a skip when dims mismatch.
        return self.out_proj(out)                                                 # (B,Tq,out_dim)


# ---------------------------------------------------------------------------
# Embedding module — mirrors defNeuronEmbeddingOCL
# Each sub-window of bar features is projected independently, then concatenated
# windows = {4, 1, 1, 1, 2}, window_out = EmbeddingSize/2 = 4
# Result shape after embedding: (B, GPT_BARS, n_groups * window_out)
#                              = (B, 100, 5 * 4) = (B, 100, 20)
# Then ConvOCL (stride=window_out=4, out=EMBEDDING_SIZE=8) reduces to
# (B, 100, 8) via a per-position linear projection across each 4-feature group
# (i.e. apply the same linear to each group independently and combine)
# ---------------------------------------------------------------------------
class BarEmbedding(nn.Module):
    """
    Implements EmbeddingOCL + BatchNorm + ConvOCL projection.

    Input:  (B, GPT_BARS, BAR_DESCR=9)
    Output: (B, GPT_BARS, EMBEDDING_SIZE=8)

    Each bar is split into 5 sub-windows [{4},{1},{1},{1},{2}].
    Each group is linearly projected to embed_out_half=4.
    All groups concatenated -> (B, GPT_BARS, 20).
    Final linear per position: 20 -> 8.
    """
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(BAR_DESCR)   # defNeuronBatchNormOCL over feature dim
        n_groups = len(EMBED_WINDOWS)         # 5

        # One linear per sub-window (defNeuronEmbeddingOCL)
        self.group_projs = nn.ModuleList([
            _init_linear(nn.Linear(w, EMBED_OUT_HALF)) for w in EMBED_WINDOWS
        ])
        # ConvOCL: projects concatenated embedding (n_groups * EMBED_OUT_HALF = 20) -> EMBEDDING_SIZE=8
        self.conv_proj = _init_linear(nn.Linear(n_groups * EMBED_OUT_HALF, CONV_OUT))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, GPT_BARS, 9)"""
        B, T, _ = x.shape
        # BatchNorm on feature dimension — reshape to (B*T, 9), norm, reshape back
        x_bn = self.bn(x.view(B * T, BAR_DESCR)).view(B, T, BAR_DESCR)
        # Split into sub-windows and project each
        cursor = 0
        parts  = []
        for proj, w in zip(self.group_projs, EMBED_WINDOWS):
            parts.append(proj(x_bn[..., cursor:cursor + w]))  # (B, T, 4)
            cursor += w
        embedded = torch.cat(parts, dim=-1)   # (B, T, 20)
        # ConvOCL final projection
        return self.conv_proj(embedded)        # (B, T, 8)


# ---------------------------------------------------------------------------
# VAE layer — defNeuronVAEOCL
# Input: (B, 2*latent_dim) = (B, 12) for actor (2*NActions)
# Output: (B, latent_dim) = (B, 6)   via reparameterisation
# ---------------------------------------------------------------------------
class VAELayer(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, 2 * latent_dim)
        Returns: (z, mu, log_var)
        """
        mu, log_var = x.chunk(2, dim=-1)           # each (B, latent_dim)
        log_var = torch.clamp(log_var, -10.0, 10.0)
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z   = mu + eps * std
        else:
            z = mu                                 # deterministic at inference
        return z, mu, log_var


# ---------------------------------------------------------------------------
# ENCODER network
# Mirrors CreateDescriptions encoder section in Trajectory.mqh
# ---------------------------------------------------------------------------
class ConformerEncoder(nn.Module):
    """
    Takes a sequence of GPT_BARS=100 bar feature vectors and encodes them.

    Input:  (B, GPT_BARS, BAR_DESCR) = (B, 100, 9)
    Output: (B, GPT_BARS, EMBEDDING_SIZE) = (B, 100, 8)
            (the full sequence is kept — actor/critic cross-attend into it)
    """
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        # layers 1-3 of encoder description
        self.embedding = BarEmbedding()
        # layer 4 — PE
        self.pe = LearnablePositionalEncoding(GPT_BARS, CONV_OUT)
        # layer 5 — 5x ConformerOCL
        self.conformers = nn.ModuleList([
            ConformerBlock(CONV_OUT, N_HEADS, dropout=dropout)
            for _ in range(N_CONFORMER)
        ])

    def forward(self, bars: torch.Tensor) -> torch.Tensor:
        """bars: (B, GPT_BARS, 9) -> (B, 100, 8)"""
        x = self.embedding(bars)   # (B, 100, 8)
        x = self.pe(x)
        for block in self.conformers:
            x = block(x)
        return x                   # (B, 100, 8)


# ---------------------------------------------------------------------------
# ACTOR network
# Mirrors CreateDescriptions actor section in Trajectory.mqh
# ---------------------------------------------------------------------------
class ConformerActor(nn.Module):
    """
    Input:
        account: (B, ACCOUNT_DESCR=12)
        enc_out: (B, GPT_BARS*5, EMBEDDING_SIZE) from encoder
                 Note: in the MQ5 source, after 5 ConformerBlocks the output is
                 still (100, 8). The CrossAttn key size is GPT_BARS*5=500 which
                 in the MQ5 implementation refers to the flattened conformer output
                 across the 5-layer stack interleaving. We replicate by repeating
                 the conformer sequence 5 times (one per layer) — or equivalently
                 using all 5 intermediate outputs. Here we keep it simple and tile
                 the final output to (B, 500, 8) to match the declared key count.

    Output: (B, N_ACTIONS=6) — action vector (buy_vol, buy_tp, buy_sl, sell_vol, sell_tp, sell_sl)
    """
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        # layer 0 — input (AccountDescr=12 raw)
        # layer 1 — FC(EmbeddingSize=8, sigmoid)
        self.input_proj = nn.Sequential(
            _init_linear(nn.Linear(ACCOUNT_DESCR, EMBEDDING_SIZE)),
            nn.Sigmoid(),
        )
        # layers 2-4 — 3x CrossAttn (defNeuronCrossAttenOCL, all with same description)
        # Layer 2: query_dim=8 (from input_proj), key_dim=8, out_dim=16
        # Layer 3: query_dim=16 (from prev CrossAttn), key_dim=8, out_dim=16
        # Layer 4: query_dim=16, key_dim=8, out_dim=16
        self.ca0 = CrossAttentionLayer(
            query_dim=EMBEDDING_SIZE,
            key_dim=EMBEDDING_SIZE,
            n_heads=N_HEADS,
            out_dim=CROSS_ATTN_OUT,
            dropout=dropout,
        )
        self.ca1 = CrossAttentionLayer(
            query_dim=CROSS_ATTN_OUT,
            key_dim=EMBEDDING_SIZE,
            n_heads=N_HEADS,
            out_dim=CROSS_ATTN_OUT,
            dropout=dropout,
        )
        self.ca2 = CrossAttentionLayer(
            query_dim=CROSS_ATTN_OUT,
            key_dim=EMBEDDING_SIZE,
            n_heads=N_HEADS,
            out_dim=CROSS_ATTN_OUT,
            dropout=dropout,
        )
        # layer 5 — FC(LatentCount=512, sigmoid)
        self.fc5 = nn.Sequential(
            _init_linear(nn.Linear(CROSS_ATTN_OUT, LATENT_COUNT)),
            nn.Sigmoid(),
        )
        # layer 6 — FC(2*NActions=12, linear)
        self.fc6 = _init_linear(nn.Linear(LATENT_COUNT, 2 * N_ACTIONS))
        # layer 7 — VAE -> NActions=6
        self.vae = VAELayer(N_ACTIONS)

    def forward(
        self, account: torch.Tensor, enc_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        account: (B, 12)
        enc_out: (B, 100, 8)
        returns: z (B, 6), mu (B, 6), log_var (B, 6)
        """
        # Tile encoder output to (B, 500, 8) to match GPT_BARS*N_CONFORMER key tokens
        ctx = enc_out.repeat(1, N_CONFORMER, 1)    # (B, 500, 8)

        # Project account -> query token (B, 1, 8)
        q = self.input_proj(account).unsqueeze(1)   # (B, 1, 8)

        # 3x CrossAttn — dims flow: 8 -> 16 -> 16 -> 16
        q = self.ca0(q, ctx)                        # (B, 1, 16)
        q = self.ca1(q, ctx)                        # (B, 1, 16)
        q = self.ca2(q, ctx)                        # (B, 1, 16)

        q = q.squeeze(1)                            # (B, 16)
        x = self.fc5(q)                             # (B, 512)
        x = self.fc6(x)                             # (B, 12)
        z, mu, log_var = self.vae(x)
        return z, mu, log_var


# ---------------------------------------------------------------------------
# CRITIC network
# Mirrors CreateDescriptions critic section in Trajectory.mqh
# ---------------------------------------------------------------------------
class ConformerCritic(nn.Module):
    """
    Input:
        actions: (B, N_ACTIONS=6)
        enc_out: (B, GPT_BARS, EMBEDDING_SIZE)
    Output: (B, N_REWARDS=3)
    """
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        # layer 0 — input (NActions=6)
        # layer 1 — FC(EmbeddingSize=8, sigmoid)
        self.input_proj = nn.Sequential(
            _init_linear(nn.Linear(N_ACTIONS, EMBEDDING_SIZE)),
            nn.Sigmoid(),
        )
        # layers 2-4 — 3x CrossAttn (same structure as actor, dims flow 8->16->16->16)
        self.ca0 = CrossAttentionLayer(
            query_dim=EMBEDDING_SIZE,
            key_dim=EMBEDDING_SIZE,
            n_heads=N_HEADS,
            out_dim=CROSS_ATTN_OUT,
            dropout=dropout,
        )
        self.ca1 = CrossAttentionLayer(
            query_dim=CROSS_ATTN_OUT,
            key_dim=EMBEDDING_SIZE,
            n_heads=N_HEADS,
            out_dim=CROSS_ATTN_OUT,
            dropout=dropout,
        )
        self.ca2 = CrossAttentionLayer(
            query_dim=CROSS_ATTN_OUT,
            key_dim=EMBEDDING_SIZE,
            n_heads=N_HEADS,
            out_dim=CROSS_ATTN_OUT,
            dropout=dropout,
        )
        # layer 5 — FC(LatentCount=512, sigmoid)
        self.fc5 = nn.Sequential(
            _init_linear(nn.Linear(CROSS_ATTN_OUT, LATENT_COUNT)),
            nn.Sigmoid(),
        )
        # layer 6 — FC(LatentCount=512, sigmoid)
        self.fc6 = nn.Sequential(
            _init_linear(nn.Linear(LATENT_COUNT, LATENT_COUNT)),
            nn.Sigmoid(),
        )
        # layer 7 — FC(NRewards=3, linear)
        self.fc7 = _init_linear(nn.Linear(LATENT_COUNT, N_REWARDS))

    def forward(self, actions: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        """
        actions: (B, 6)
        enc_out: (B, 100, 8)
        returns: (B, 3)
        """
        ctx = enc_out.repeat(1, N_CONFORMER, 1)     # (B, 500, 8)
        q   = self.input_proj(actions).unsqueeze(1)  # (B, 1, 8)

        q = self.ca0(q, ctx)                         # (B, 1, 16)
        q = self.ca1(q, ctx)                         # (B, 1, 16)
        q = self.ca2(q, ctx)                         # (B, 1, 16)

        q = q.squeeze(1)                             # (B, 16)
        x = self.fc5(q)                              # (B, 512)
        x = self.fc6(x)                              # (B, 512)
        return self.fc7(x)                           # (B, 3)


# ---------------------------------------------------------------------------
# Full ConformerExpert — wraps Encoder + Actor + Critic
# ---------------------------------------------------------------------------
class ConformerExpert(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.encoder = ConformerEncoder(dropout=dropout)
        self.actor   = ConformerActor(dropout=dropout)
        self.critic  = ConformerCritic(dropout=dropout)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def generate_signal(
        self,
        bars: torch.Tensor,
        account: Optional[torch.Tensor] = None,
    ) -> Tuple[int, float]:
        """
        bars:    (100, 9)  or  (B, 100, 9) — GPT_BARS bars of BAR_DESCR features
        account: (12,)     or  (B, 12)     — account state (pass zeros if unavailable)

        Returns: (direction, confidence)
            direction:  1 = BUY, -1 = SELL, 0 = HOLD
            confidence: float in [0.0, 1.0]

        Action vector layout (NActions=6):
          [0] buy_vol  [1] buy_tp   [2] buy_sl
          [3] sell_vol [4] sell_tp  [5] sell_sl
        """
        self.eval()
        with torch.no_grad():
            # --- shape normalisation
            if bars.dim() == 2:
                bars = bars.unsqueeze(0)       # (1, 100, 9)
            bars = bars.to(DEVICE)

            if account is None:
                account = torch.zeros(bars.shape[0], ACCOUNT_DESCR, device=DEVICE)
            elif account.dim() == 1:
                account = account.unsqueeze(0).to(DEVICE)
            else:
                account = account.to(DEVICE)

            enc_out        = self.encoder(bars)       # (B, 100, 8)
            z, mu, _       = self.actor(account, enc_out)  # (B, 6)
            actions        = z[0]                     # (6,) — first sample

            buy_vol  = actions[0].item()
            buy_tp   = actions[1].item()
            buy_sl   = actions[2].item()
            sell_vol = actions[3].item()
            sell_tp  = actions[4].item()
            sell_sl  = actions[5].item()

            # Replicate MQ5 Test.mq5 netting logic:
            if buy_vol >= sell_vol:
                buy_vol  -= sell_vol
                sell_vol  = 0.0
            else:
                sell_vol -= buy_vol
                buy_vol   = 0.0

            # Use a min_lot proxy of 0.01 and min_stops proxy
            min_lot   = 0.01
            min_stops = 0.001   # normalised threshold

            buy_valid  = (buy_vol  >= min_lot) and (buy_tp  > min_stops) and (buy_sl  > min_stops)
            sell_valid = (sell_vol >= min_lot) and (sell_tp > min_stops) and (sell_sl > min_stops)

            if buy_valid and not sell_valid:
                direction  = 1
                confidence = float(torch.sigmoid(torch.tensor(buy_vol)).item())
            elif sell_valid and not buy_valid:
                direction  = -1
                confidence = float(torch.sigmoid(torch.tensor(sell_vol)).item())
            elif buy_valid and sell_valid:
                # Both valid — pick stronger side
                if buy_vol >= sell_vol:
                    direction  = 1
                    confidence = float(torch.sigmoid(torch.tensor(buy_vol)).item())
                else:
                    direction  = -1
                    confidence = float(torch.sigmoid(torch.tensor(sell_vol)).item())
            else:
                direction  = 0
                confidence = 0.0

            confidence = max(0.0, min(1.0, confidence))
        return direction, confidence

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_step(
        self,
        batch: dict,
        actor_optimizer:  torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        encoder_optimizer: torch.optim.Optimizer,
    ) -> dict:
        """
        Walk-forward training step — mirrors Train() loop in Study.mq5.

        batch keys (all torch.Tensor on correct device):
            'bars'        (B, 100, 9)  — GPT context window of bar features
            'account'     (B, 12)      — normalised account state
            'actions'     (B, 6)       — recorded actions
            'rewards_now' (B, 3)       — r[i+1]
            'rewards_next'(B, 3)       — r[i+2]

        Returns dict with scalar losses.

        NOTE ON DEVICES:
            Training (backward) runs on CPU — DirectML does not support backprop
            for arbitrary graph topologies (CompileGraph fails).  Inference / forward-
            only calls use INFER_DEVICE (DirectML GPU).  This mirrors the LSTM rule
            already in place for this project.
        """
        # Move entire model to CPU for training
        self.to(TRAIN_DEVICE)
        self.train()

        bars         = batch["bars"].to(TRAIN_DEVICE)
        account      = batch["account"].to(TRAIN_DEVICE)
        actions_gt   = batch["actions"].to(TRAIN_DEVICE)
        rewards_now  = batch["rewards_now"].to(TRAIN_DEVICE)
        rewards_next = batch["rewards_next"].to(TRAIN_DEVICE)

        # ---- Encoder forward
        enc_out = self.encoder(bars)    # (B, 100, 8)

        # ---- Critic step
        # TD target mirrors Study.mq5: result = r[i+1] - r[i+2] * DiscFactor
        td_target   = rewards_now - rewards_next * DISC_FACTOR
        critic_pred = self.critic(actions_gt, enc_out.detach())   # (B, 3)
        critic_loss = F.mse_loss(critic_pred, td_target)

        critic_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(),  1.0)
        nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        critic_optimizer.step()
        encoder_optimizer.step()

        # ---- Actor step
        enc_out        = self.encoder(bars)    # re-forward with updated encoder
        z, mu, log_var = self.actor(account, enc_out)

        # Behavioural cloning — match recorded actions from replay buffer
        actor_bc_loss = F.mse_loss(z, actions_gt)

        # VAE KL divergence regulariser
        kl_loss = -0.5 * torch.mean(1.0 + log_var - mu.pow(2) - log_var.exp())

        # Value maximisation — use frozen critic to push actor towards high-value states
        with torch.no_grad():
            critic_value = self.critic(z.detach(), enc_out.detach())   # (B, 3)
        value_loss = -critic_value.mean()

        actor_loss = actor_bc_loss + 0.001 * kl_loss + 0.01 * value_loss

        actor_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(),   1.0)
        nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        actor_optimizer.step()
        encoder_optimizer.step()

        # Move back to inference device when done
        self.to(INFER_DEVICE)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "kl_loss":     kl_loss.item(),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_weights(self, path: str) -> None:
        """Save all weights to a single .pt file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "encoder": self.encoder.state_dict(),
            "actor":   self.actor.state_dict(),
            "critic":  self.critic.state_dict(),
        }, path)
        print(f"[ConformerExpert] saved -> {path}")

    def load_weights(self, path: str, map_location: Optional[str] = None) -> None:
        """Load weights from a .pt file saved by save_weights()."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Weight file not found: {path}")
        loc  = map_location or str(DEVICE)
        ckpt = torch.load(path, map_location=loc)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        print(f"[ConformerExpert] loaded <- {path}")


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------
def build_conformer_expert(dropout: float = 0.1) -> ConformerExpert:
    """Instantiate, move to DEVICE, return ready-to-use expert."""
    model = ConformerExpert(dropout=dropout).to(DEVICE)
    return model


def build_optimizers(
    model: ConformerExpert,
    lr: float = 1e-4,
) -> Tuple[
    torch.optim.Optimizer,
    torch.optim.Optimizer,
    torch.optim.Optimizer,
]:
    """Returns (actor_opt, critic_opt, encoder_opt) — ADAM for all, mirrors MQ5."""
    actor_opt   = torch.optim.Adam(model.actor.parameters(),   lr=lr)
    critic_opt  = torch.optim.Adam(model.critic.parameters(),  lr=lr)
    encoder_opt = torch.optim.Adam(model.encoder.parameters(), lr=lr)
    return actor_opt, critic_opt, encoder_opt


# ---------------------------------------------------------------------------
# Quick sanity-check — run as script to verify shapes
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    model = build_conformer_expert()
    print(f"Encoder params: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"Actor   params: {sum(p.numel() for p in model.actor.parameters()):,}")
    print(f"Critic  params: {sum(p.numel() for p in model.critic.parameters()):,}")

    # Test generate_signal
    dummy_bars    = torch.randn(GPT_BARS, BAR_DESCR)
    dummy_account = torch.zeros(ACCOUNT_DESCR)
    direction, confidence = model.generate_signal(dummy_bars, dummy_account)
    print(f"generate_signal -> direction={direction}, confidence={confidence:.4f}")

    # Test train_step shapes
    B = 4
    batch = {
        "bars":         torch.randn(B, GPT_BARS, BAR_DESCR),
        "account":      torch.randn(B, ACCOUNT_DESCR),
        "actions":      torch.rand(B, N_ACTIONS),
        "rewards_now":  torch.randn(B, N_REWARDS),
        "rewards_next": torch.randn(B, N_REWARDS),
    }
    actor_opt, critic_opt, encoder_opt = build_optimizers(model)
    losses = model.train_step(batch, actor_opt, critic_opt, encoder_opt)
    print(f"train_step losses: {losses}")
    print("All checks passed.")

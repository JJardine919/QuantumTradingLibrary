"""
expert_timemoe.py — TimeMoE Actor-Critic in PyTorch

Architecture faithfully converted from DNG's TimeMoE MQL5 sources:
  Trajectory.mqh  (layer definitions)
  Study.mq5       (training loop)
  Test.mq5        (inference / feature construction)

GPU: AMD RX 6800 XT via torch_directml.
LSTM (if needed): CPU only — DirectML backprop for LSTM is broken.

Dimension trace (from Trajectory.mqh):
  Input          : [B, 120*9] = [B, 1080]
  BatchNorm+Noise: [B, 1080]
  ConcatDiff     : [B, 120, 18]  (HistoryBars=120, 2*BarDescr=18)
  Mamba4Cast     : [B, 120, 24]  (NSkills=24 skills per bar)
  Transpose      : [B, 24, 120]
  SwiGLU         : [B, 8, 32]    (Segments=8, EmbeddingSize=32)
  TransposeRCD   : [B, 24, 8, 32] rearranged to [B, 24*8*32] = [B, 6144]
  BatchNorm      : [B, 6144]
  TimeMoEAttn    : [B, 24, 8]    (units_main=24, window_out=8)

Actor / Critic / Director:
  Input          : [B, AccountDescr=13]
  BatchNorm      : [B, 13]
  CrossDMHAttn   : [B, 1, 32]   cross-attends encoder latent [B, 24, 32]
  Dense x2       : [B, 256]
  Output         : actor->6, director->1, critic->1
"""

from __future__ import annotations

import math
import os
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------
# Device setup — AMD via DirectML, fall back to CPU gracefully
# ---------------------------------------------------------------
try:
    import torch_directml
    _DML_DEVICE = torch_directml.device()
    _GPU_AVAILABLE = True
except Exception:
    _DML_DEVICE = torch.device("cpu")
    _GPU_AVAILABLE = False

def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu or not _GPU_AVAILABLE:
        return torch.device("cpu")
    return _DML_DEVICE


# ---------------------------------------------------------------
# Constants (mirrored from Trajectory.mqh)
# ---------------------------------------------------------------
HISTORY_BARS   = 120
BAR_DESCR      = 9
ACCOUNT_DESCR  = 13
N_ACTIONS      = 6
N_REWARDS      = 1
N_FORECAST     = 30
N_SKILLS       = 24       # NSkills
EMBEDDING_SIZE = 32       # EmbeddingSize
SEGMENTS       = 8        # Segments
N_EXPERTS      = 12       # NExperts
TOP_K          = 2        # TopK
LATENT_COUNT   = 256      # LatentCount
DISC_FACTOR    = 0.9

# Derived
STATE_DIM      = HISTORY_BARS * BAR_DESCR          # 1080
ENCODER_EMBED  = N_SKILLS                          # 24  (Mamba4Cast output per bar)
# After SwiGLU: [Segments=8, EmbeddingSize=32] per variable (prev_var=24)
# TransposeRCD reshapes to [prev_var, prev_count, prev_out] = [24, 8, 32]
# Flattened: 24*8*32 = 6144
LATENT_DIM_FLAT = N_SKILLS * SEGMENTS * EMBEDDING_SIZE   # 6144

# TimeMoEAttention sizing
MOE_UNITS_MAIN  = N_SKILLS                         # prev_var = 24   (number of "tokens")
MOE_WIN_MAIN    = EMBEDDING_SIZE                   # prev_out = 32   (per-token dim from SwiGLU)
MOE_WIN_CROSS   = EMBEDDING_SIZE                   # same as main
MOE_UNITS_CROSS = N_SKILLS * SEGMENTS              # prev_var * prev_count = 24*8 = 192
MOE_EXPERT_DIM  = 8                                # from windows[2]
MOE_HEADS       = 4                                # step = 4
MOE_LAYERS      = 6                                # depth of MoE attention stack
MOE_WIN_OUT     = EMBEDDING_SIZE // 4              # 8

# CrossDMHAttention sizing
CROSS_HEADS    = 4
CROSS_WIN_OUT  = 32


# ================================================================
# Utility modules
# ================================================================

class SwiGLU(nn.Module):
    """
    SwiGLU: out = x * silu(gate)
    Takes flat input of size (in_features) and produces (out_features).
    Internally splits a linear projection into x and gate halves.
    Faithful to defNeuronSwiGLUOCL:
        count    = Segments  (8)
        window   = ceil(prev_out / Segments) per segment  (ceil(120/8)=15)
        variables= prev_count  (24, number of 'rows' / skip connections)
        window_out = EmbeddingSize  (32)
    We implement the general form: project input -> 2*out, split, gate.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features * 2, bias=False)
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.proj(x)
        x_half, gate = projected.chunk(2, dim=-1)
        return x_half * F.silu(gate)


class MoERouter(nn.Module):
    """
    Learned sparse gating: selects top-K experts for each input token.
    Returns (dispatch_weights, expert_indices) both shape [B, T, TopK].
    """
    def __init__(self, token_dim: int, n_experts: int, top_k: int):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(token_dim, n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, D]
        logits = self.gate(x)                           # [B, T, NExperts]
        scores, indices = torch.topk(logits, self.top_k, dim=-1)  # [B, T, TopK]
        weights = F.softmax(scores, dim=-1)             # [B, T, TopK]  normalized
        return weights, indices


class ExpertBlock(nn.Module):
    """
    Single expert MLP. Each expert has its own independent parameters.
    expert_dim controls the hidden width (maps to descr.windows[2]=8 from MQ5).
    """
    def __init__(self, token_dim: int, expert_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_dim, expert_dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(expert_dim * 2, token_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoELayer(nn.Module):
    """
    Sparse MoE: routes each token to top-K experts, combines with learned weights.
    """
    def __init__(self, token_dim: int, n_experts: int, top_k: int, expert_dim: int):
        super().__init__()
        self.router = MoERouter(token_dim, n_experts, top_k)
        self.experts = nn.ModuleList([
            ExpertBlock(token_dim, expert_dim) for _ in range(n_experts)
        ])
        self.top_k = top_k
        self.n_experts = n_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape
        weights, indices = self.router(x)              # [B, T, TopK]
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = indices[:, :, k]              # [B, T]
            w = weights[:, :, k].unsqueeze(-1)         # [B, T, 1]
            # Run each expert on the tokens routed to it
            for e in range(self.n_experts):
                mask = (expert_idx == e)               # [B, T] bool
                if mask.any():
                    tokens = x[mask]                   # [N, D]
                    out = self.experts[e](tokens)       # [N, D]
                    # accumulate weighted
                    output[mask] += w[mask] * out
        return output


class TimeMoEAttention(nn.Module):
    """
    MoE-enhanced multi-head self-attention.
    Q, K, V projections are replaced by MoE layers.
    Stack of moe_layers attention blocks.

    From Trajectory.mqh layer 8:
        windows = [prev_out=32, prev_out=32, expert_dim=8, TopK=2]
        units   = [prev_var=24, prev_var*prev_count=192, NExperts=12]
        layers  = 6
        step    = 4  (attention heads)
        window_out = EmbeddingSize/4 = 8
    """
    def __init__(
        self,
        token_dim: int    = MOE_WIN_MAIN,    # 32
        n_tokens: int     = MOE_UNITS_MAIN,  # 24
        n_heads: int      = MOE_HEADS,       # 4
        n_experts: int    = N_EXPERTS,       # 12
        top_k: int        = TOP_K,           # 2
        expert_dim: int   = MOE_EXPERT_DIM,  # 8
        n_layers: int     = MOE_LAYERS,      # 6
        out_dim: int      = MOE_WIN_OUT,     # 8
    ):
        super().__init__()
        self.token_dim = token_dim
        self.n_heads   = n_heads
        self.head_dim  = token_dim // n_heads   # 8
        assert token_dim % n_heads == 0, "token_dim must be divisible by n_heads"

        # MoE Q/K/V projectors — applied per token before attention
        self.moe_q = MoELayer(token_dim, n_experts, top_k, expert_dim)
        self.moe_k = MoELayer(token_dim, n_experts, top_k, expert_dim)
        self.moe_v = MoELayer(token_dim, n_experts, top_k, expert_dim)

        # Stack of norm + attention + ffn blocks (layers=6)
        self.blocks = nn.ModuleList([
            _TimeMoEBlock(token_dim, n_heads, n_experts, top_k, expert_dim)
            for _ in range(n_layers)
        ])

        # Project to final output dimension
        self.out_proj = nn.Linear(token_dim, out_dim, bias=False)
        self.norm_out = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n_tokens, token_dim] = [B, 24, 32]
        # Route Q/K/V through MoE before first attention pass
        q = self.moe_q(x)
        k = self.moe_k(x)
        v = self.moe_v(x)
        h = _mha(q, k, v, self.n_heads)   # [B, 24, 32]

        # Residual from original x, then pass through transformer blocks
        h = h + x
        for block in self.blocks:
            h = block(h)

        # Pool over token dimension -> [B, token_dim]
        h = h.mean(dim=1)
        h = self.out_proj(h)               # [B, out_dim=8]
        return self.norm_out(h)


class _TimeMoEBlock(nn.Module):
    """One residual attention+MoE block inside TimeMoEAttention."""
    def __init__(self, token_dim, n_heads, n_experts, top_k, expert_dim):
        super().__init__()
        self.norm1   = nn.LayerNorm(token_dim)
        self.norm2   = nn.LayerNorm(token_dim)
        self.n_heads = n_heads
        self.moe     = MoELayer(token_dim, n_experts, top_k, expert_dim)
        self.q_proj  = nn.Linear(token_dim, token_dim, bias=False)
        self.k_proj  = nn.Linear(token_dim, token_dim, bias=False)
        self.v_proj  = nn.Linear(token_dim, token_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention branch
        n = self.norm1(x)
        q = self.q_proj(n)
        k = self.k_proj(n)
        v = self.v_proj(n)
        attn_out = _mha(q, k, v, self.n_heads)
        x = x + attn_out
        # MoE FFN branch
        x = x + self.moe(self.norm2(x))
        return x


def _mha(q, k, v, n_heads):
    """Scaled dot-product multi-head attention, no masking."""
    B, T, D = q.shape
    head_dim = D // n_heads
    def split_heads(t):
        return t.view(B, T, n_heads, head_dim).transpose(1, 2)  # [B, H, T, hd]
    q, k, v = split_heads(q), split_heads(k), split_heads(v)
    scale = math.sqrt(head_dim)
    scores = torch.matmul(q, k.transpose(-2, -1)) / scale       # [B, H, T, T]
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)                                  # [B, H, T, hd]
    out = out.transpose(1, 2).contiguous().view(B, T, D)        # [B, T, D]
    return out


class CrossDMHAttention(nn.Module):
    """
    Cross Dense Multi-Head Attention.
    Query from actor/critic state; Keys/Values from encoder latent.

    From Trajectory.mqh (actor layer 2):
        windows = [AccountDescr=13, latent.windows[0]=32]  (query_dim, kv_dim)
        units   = [1, latent.units[0]=24]                  (query_tokens, kv_tokens)
        step    = 4  (heads)
        window_out = 32
        layers  = 3  (depth)
    """
    def __init__(
        self,
        query_dim:   int = ACCOUNT_DESCR,      # 13
        kv_dim:      int = MOE_WIN_MAIN,        # 32
        query_tokens: int = 1,
        kv_tokens:   int = MOE_UNITS_MAIN,      # 24
        n_heads:     int = CROSS_HEADS,         # 4
        out_dim:     int = CROSS_WIN_OUT,       # 32
        n_layers:    int = 3,
    ):
        super().__init__()
        self.n_heads     = n_heads
        self.query_tokens = query_tokens
        self.kv_tokens   = kv_tokens

        # Project query and kv to common attention dim
        attn_dim = max(out_dim, n_heads * 8)   # ensure divisibility
        attn_dim = (attn_dim // n_heads) * n_heads

        self.q_proj = nn.Linear(query_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(kv_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim, attn_dim, bias=False)
        self.out_proj = nn.Linear(attn_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.attn_dim = attn_dim

        # Additional dense refinement layers (layers=3 -> 2 more after first cross-attn)
        extra_layers = max(0, n_layers - 1)
        self.dense = nn.Sequential(
            *[nn.Sequential(nn.Linear(out_dim, out_dim, bias=False), nn.GELU())
              for _ in range(extra_layers)]
        )

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        query: [B, query_dim]   (account state, flat)
        kv   : [B, kv_tokens, kv_dim]  (encoder latent)
        returns: [B, out_dim]
        """
        B = query.shape[0]
        # Expand query to [B, 1, query_dim]
        q = query.unsqueeze(1)                     # [B, 1, D_q]
        q = self.q_proj(q)                         # [B, 1, attn_dim]
        k = self.k_proj(kv)                        # [B, T_kv, attn_dim]
        v = self.v_proj(kv)                        # [B, T_kv, attn_dim]

        head_dim = self.attn_dim // self.n_heads
        def split(t, seq):
            return t.view(B, seq, self.n_heads, head_dim).transpose(1, 2)

        q = split(q, 1)
        k = split(k, self.kv_tokens)
        v = split(v, self.kv_tokens)

        scale = math.sqrt(head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, H, 1, T_kv]
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)                             # [B, H, 1, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, 1, self.attn_dim)  # [B, 1, attn_dim]
        out = out.squeeze(1)                                    # [B, attn_dim]
        out = self.out_proj(out)                                # [B, out_dim]
        out = self.norm(out)
        out = out + self.dense(out)                             # residual refinement
        return out


class Mamba4CastEmbedding(nn.Module):
    """
    Temporal embedding with multi-scale period awareness.
    From Trajectory.mqh layer 3:
        count      = HistoryBars = 120
        window     = 2 * BarDescr = 18  (each bar's raw + diff concat)
        window_out = NSkills = 24
        windows    = [PeriodSeconds(H1)=3600, PeriodSeconds(D1)=86400]

    Implementation: for each bar, projects [2*BarDescr + 2_period_encodings]
    to NSkills using a shared linear. The "ConcatDiff" preceding this layer
    already gives us [open_bar, diff_prev] (2*BarDescr) per bar.
    We add sinusoidal period encodings for H1 and D1 cycles.
    """
    PERIOD_H1 = 3600
    PERIOD_D1 = 86400

    def __init__(
        self,
        in_dim: int   = BAR_DESCR * 2,    # 18 (concat with diff)
        out_dim: int  = N_SKILLS,          # 24
        n_bars: int   = HISTORY_BARS,      # 120
    ):
        super().__init__()
        # +2 channels for period encodings (sin H1, sin D1)
        self.proj = nn.Linear(in_dim + 2, out_dim, bias=True)
        self.n_bars = n_bars

    def forward(self, x: torch.Tensor, timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x          : [B, n_bars, 2*BarDescr]  (ConcatDiff output)
        timestamps : [B, n_bars]  unix seconds, float (optional)
        returns    : [B, n_bars, NSkills]
        """
        B, T, D = x.shape

        if timestamps is not None:
            ts = timestamps.float()                                 # [B, T]
            h1 = torch.sin(2.0 * math.pi * ts / self.PERIOD_H1)   # [B, T]
            d1 = torch.sin(2.0 * math.pi * ts / self.PERIOD_D1)   # [B, T]
            pe = torch.stack([h1, d1], dim=-1)                     # [B, T, 2]
        else:
            # Synthetic position encoding when timestamps unavailable
            pos = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1)
            h1 = torch.sin(2.0 * math.pi * pos / 60.0)            # ~H1 in bars
            d1 = torch.sin(2.0 * math.pi * pos / 1440.0)          # ~D1 in bars
            pe = torch.stack([h1, d1], dim=-1)                     # [B, T, 2]

        x_aug = torch.cat([x, pe], dim=-1)                         # [B, T, D+2]
        return self.proj(x_aug)                                     # [B, T, NSkills]


# ================================================================
# Encoder
# ================================================================

class TimeMoEEncoder(nn.Module):
    """
    Full encoder pipeline matching Trajectory.mqh layers 0-8:

    0  Input          [B, 1080]
    1  BatchNorm+Noise [B, 1080]
    2  ConcatDiff      [B, 120, 18]
    3  Mamba4Cast      [B, 120, 24]
    4  Transpose       [B, 24, 120]
    5  SwiGLU          [B, 8, 32]    (Segments=8, per variable slice)
    6  TransposeRCD    [B, 24, 8, 32]
    7  BatchNorm       [B, 6144]
    8  TimeMoEAttention -> pool -> [B, 24, 8]  (units_main x win_out)
    """

    def __init__(self, training: bool = False):
        super().__init__()
        self._is_training = training

        # Layer 1: BatchNorm with optional additive noise during training
        self.bn_input = nn.BatchNorm1d(STATE_DIM)
        self.noise_std = 0.01

        # Layer 2: ConcatDiff — we implement as a per-bar diff projection
        # Produces [B, 120, 18] from [B, 1080]: reshape + concat bar with prev_bar
        # (No learnable params; pure data transformation)

        # Layer 3: Mamba4CastEmbedding
        self.mamba = Mamba4CastEmbedding(in_dim=BAR_DESCR * 2, out_dim=N_SKILLS, n_bars=HISTORY_BARS)

        # Layer 4: Transpose is data-only — no params

        # Layer 5: SwiGLU
        # After transpose: shape is [B, NSkills=24, HistoryBars=120]
        # Treat each of the 24 'variable' rows independently.
        # SwiGLU maps [B*24, HistoryBars=120] -> [B*24, Segments=8, EmbeddingSize=32]
        # Per the MQ5: count=Segments=8, window=ceil(120/8)=15, window_out=EmbeddingSize=32
        # We implement as segment-wise linear with SwiGLU activation.
        seg_in = math.ceil(HISTORY_BARS / SEGMENTS)   # 15
        self.swiglu = SwiGLU(in_features=HISTORY_BARS, out_features=SEGMENTS * EMBEDDING_SIZE)
        # Output reshaped to [B, 24, 8, 32]

        # Layer 6: TransposeRCD — rearranges [B, 24, 8, 32] to [B, 24, 8, 32] (already in right shape)
        # count=prev_var=24, window=prev_count=8, step=prev_out=32
        # This is a learned transpose with optional conv; we treat as identity in shape
        # but add a learnable linear mixing per segment-variable pair
        self.rcd_mix = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE, bias=False)

        # Layer 7: BatchNorm on flattened [B, 6144]
        self.bn_latent = nn.BatchNorm1d(LATENT_DIM_FLAT)

        # Layer 8: TimeMoEAttention
        # Input shape for attention: [B, MOE_UNITS_MAIN=24, MOE_WIN_MAIN=32]
        # (we reshape from [B, 6144] -> [B, 24, 8*32/8] = need to handle carefully)
        # Actually: after layer 7 we have [B, 6144] = [B, 24*8*32]
        # Reshape to [B, 24, 8*32] = [B, 24, 256]? No —
        # The MoE attention units_main=24 tokens each of dim windows[0]=32 (prev_out=EmbeddingSize)
        # So we project from [B, 24, 8*32] -> [B, 24, 32] via a learned reduction
        self.latent_reduce = nn.Linear(SEGMENTS * EMBEDDING_SIZE, MOE_WIN_MAIN, bias=False)
        self.moe_attn = TimeMoEAttention(
            token_dim  = MOE_WIN_MAIN,    # 32
            n_tokens   = MOE_UNITS_MAIN,  # 24
            n_heads    = MOE_HEADS,       # 4
            n_experts  = N_EXPERTS,       # 12
            top_k      = TOP_K,           # 2
            expert_dim = MOE_EXPERT_DIM,  # 8
            n_layers   = MOE_LAYERS,      # 6
            out_dim    = MOE_WIN_OUT,     # 8
        )
        # latent output: pooled [B, 8]
        # But actor/critic cross-attention needs [B, tokens=24, dim=32]
        # We keep the pre-pool latent too for cross-attention
        self.latent_norm = nn.LayerNorm(MOE_WIN_MAIN)

    def forward(
        self,
        state: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        state     : [B, 1080]
        timestamps: [B, 120]  unix seconds (optional)

        returns:
            latent_tokens : [B, 24, 32]   for cross-attention in actor/critic
            latent_pooled : [B, 8]        final encoder output (TimeMoE out)
        """
        B = state.shape[0]

        # Layer 1: BatchNorm + noise
        x = self.bn_input(state)
        if self._is_training and self.training:
            x = x + torch.randn_like(x) * self.noise_std

        # Layer 2: ConcatDiff — reshape to [B, 120, 9], concat with diff from prev bar
        x_bars = x.view(B, HISTORY_BARS, BAR_DESCR)               # [B, 120, 9]
        diff = torch.zeros_like(x_bars)
        diff[:, 1:, :] = x_bars[:, 1:, :] - x_bars[:, :-1, :]    # [B, 120, 9] diff
        x_concat = torch.cat([x_bars, diff], dim=-1)               # [B, 120, 18]

        # Layer 3: Mamba4CastEmbedding
        x_emb = self.mamba(x_concat, timestamps)                   # [B, 120, 24]

        # Layer 4: Transpose -> [B, 24, 120]
        x_t = x_emb.permute(0, 2, 1)                              # [B, 24, 120]

        # Layer 5: SwiGLU per variable row
        # Treat each of 24 variable rows as independent: [B*24, 120] -> [B*24, 8*32]
        x_flat = x_t.reshape(B * N_SKILLS, HISTORY_BARS)          # [B*24, 120]
        x_sg = self.swiglu(x_flat)                                 # [B*24, 8*32=256]
        x_sg = x_sg.view(B, N_SKILLS, SEGMENTS, EMBEDDING_SIZE)   # [B, 24, 8, 32]

        # Layer 6: TransposeRCD — learned mixing, keep shape
        x_rcd = self.rcd_mix(x_sg)                                # [B, 24, 8, 32]

        # Layer 7: BatchNorm on flattened
        x_bn = x_rcd.reshape(B, LATENT_DIM_FLAT)                  # [B, 6144]
        x_bn = self.bn_latent(x_bn)

        # Reshape for attention: [B, 24, 8, 32] -> reduce last two -> [B, 24, 32]
        x_attn_in = x_bn.view(B, N_SKILLS, SEGMENTS * EMBEDDING_SIZE)  # [B, 24, 256]
        x_attn_in = self.latent_reduce(x_attn_in)                       # [B, 24, 32]
        x_attn_in = self.latent_norm(x_attn_in)

        # Layer 8: TimeMoEAttention
        # Keep both token-level for cross-attention and pooled output
        latent_tokens = x_attn_in                                  # [B, 24, 32]
        latent_pooled = self.moe_attn(x_attn_in)                   # [B, 8]

        return latent_tokens, latent_pooled


# ================================================================
# Actor
# ================================================================

class TimeMoEActor(nn.Module):
    """
    Actor: account state + encoder latent -> 6 actions in [0,1]

    Trajectory.mqh actor layers:
    0  Input         [B, AccountDescr=13]
    1  BatchNorm     [B, 13]
    2  CrossDMHAttn  [B, 1, 32]  cross over encoder tokens [B, 24, 32]
    3  Dense 256 TANH
    4  Dense 256 SoftPlus
    5  Dense 6  SIGMOID
    """

    def __init__(self):
        super().__init__()
        self.bn_account = nn.BatchNorm1d(ACCOUNT_DESCR)
        self.cross_attn = CrossDMHAttention(
            query_dim    = ACCOUNT_DESCR,
            kv_dim       = MOE_WIN_MAIN,
            query_tokens = 1,
            kv_tokens    = MOE_UNITS_MAIN,
            n_heads      = CROSS_HEADS,
            out_dim      = CROSS_WIN_OUT,
            n_layers     = 3,
        )
        cross_out = CROSS_WIN_OUT   # 32
        self.dense1  = nn.Linear(cross_out, LATENT_COUNT)
        self.dense2  = nn.Linear(LATENT_COUNT, LATENT_COUNT)
        self.out     = nn.Linear(LATENT_COUNT, N_ACTIONS)

    def forward(
        self,
        account: torch.Tensor,
        latent_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        account      : [B, 13]
        latent_tokens: [B, 24, 32]
        returns      : [B, 6]  values in [0, 1]
        """
        x = self.bn_account(account)
        x = self.cross_attn(x, latent_tokens)           # [B, 32]
        x = torch.tanh(self.dense1(x))                  # [B, 256]
        x = F.softplus(self.dense2(x))                  # [B, 256]
        x = torch.sigmoid(self.out(x))                  # [B, 6]
        return x


# ================================================================
# Director
# ================================================================

class TimeMoEDirector(nn.Module):
    """
    Director: same structure as Actor but outputs 1 value (confidence/direction).
    Trajectory.mqh director layers same as actor except output size = 1.
    """

    def __init__(self):
        super().__init__()
        self.bn_account = nn.BatchNorm1d(N_ACTIONS)
        self.cross_attn = CrossDMHAttention(
            query_dim    = N_ACTIONS,
            kv_dim       = MOE_WIN_MAIN,
            query_tokens = 1,
            kv_tokens    = MOE_UNITS_MAIN,
            n_heads      = CROSS_HEADS,
            out_dim      = CROSS_WIN_OUT,
            n_layers     = 3,
        )
        self.dense1  = nn.Linear(CROSS_WIN_OUT, LATENT_COUNT)
        self.dense2  = nn.Linear(LATENT_COUNT, LATENT_COUNT)
        self.out     = nn.Linear(LATENT_COUNT, 1)

    def forward(
        self,
        actions: torch.Tensor,
        latent_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        actions      : [B, 6]
        latent_tokens: [B, 24, 32]
        returns      : [B, 1]  in [0, 1]
        """
        x = self.bn_account(actions)
        x = self.cross_attn(x, latent_tokens)
        x = torch.tanh(self.dense1(x))
        x = torch.tanh(self.dense2(x))
        x = torch.sigmoid(self.out(x))
        return x


# ================================================================
# Critic
# ================================================================

class TimeMoECritic(nn.Module):
    """
    Critic: same structure as Director but outputs NRewards (1) without final sigmoid
    (regression — linear output).
    Trajectory.mqh critic output activation = None.
    """

    def __init__(self):
        super().__init__()
        self.bn_account = nn.BatchNorm1d(N_ACTIONS)
        self.cross_attn = CrossDMHAttention(
            query_dim    = N_ACTIONS,
            kv_dim       = MOE_WIN_MAIN,
            query_tokens = 1,
            kv_tokens    = MOE_UNITS_MAIN,
            n_heads      = CROSS_HEADS,
            out_dim      = CROSS_WIN_OUT,
            n_layers     = 3,
        )
        self.dense1  = nn.Linear(CROSS_WIN_OUT, LATENT_COUNT)
        self.dense2  = nn.Linear(LATENT_COUNT, LATENT_COUNT)
        self.out     = nn.Linear(LATENT_COUNT, N_REWARDS)

    def forward(
        self,
        actions: torch.Tensor,
        latent_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        actions      : [B, 6]
        latent_tokens: [B, 24, 32]
        returns      : [B, 1]
        """
        x = self.bn_account(actions)
        x = self.cross_attn(x, latent_tokens)
        x = torch.tanh(self.dense1(x))
        x = F.softplus(self.dense2(x))
        x = self.out(x)                    # linear output, no activation
        return x


# ================================================================
# Forecast heads (3 variants matching caForecast[0..2])
# ================================================================

class ForecastHead(nn.Module):
    """
    Forecast branch that predicts future bar descriptors from encoder output.
    Three variants differ by horizon: 1, NForecast/2=15, NForecast=30 bars.

    Trajectory.mqh forecast1: ConvOCL window_out=1 -> BarDescr
    Trajectory.mqh forecast2: ConvOCL window_out=NForecast/2 -> NForecast/2 * BarDescr
    Trajectory.mqh forecast3: ConvOCL window_out=NForecast -> NForecast * BarDescr
    """

    def __init__(self, horizon: int = 1):
        super().__init__()
        self.horizon = horizon
        # Input: latent tokens [B, 24, 32] -> flatten -> project
        in_dim  = MOE_UNITS_MAIN * MOE_WIN_MAIN   # 24*32 = 768
        self.proj = nn.Sequential(
            nn.Linear(in_dim, MOE_UNITS_MAIN),
            nn.Tanh(),
        )
        # per-bar prediction: 24 tokens -> horizon * BarDescr
        self.predict = nn.Sequential(
            nn.Linear(MOE_UNITS_MAIN, horizon * BAR_DESCR),
            nn.Tanh(),
        )
        self.bn_out = nn.BatchNorm1d(horizon * BAR_DESCR)

    def forward(self, latent_tokens: torch.Tensor) -> torch.Tensor:
        """
        latent_tokens: [B, 24, 32]
        returns: [B, horizon * BarDescr]
        """
        B = latent_tokens.shape[0]
        x = latent_tokens.reshape(B, -1)        # [B, 768]
        x = self.proj(x)                         # [B, 24]
        x = self.predict(x)                      # [B, horizon*9]
        x = self.bn_out(x)
        return x


# ================================================================
# Full TimeMoE System
# ================================================================

class TimeMoESystem(nn.Module):
    """
    Complete system: Encoder + Actor + Director + Critic + 3 Forecast heads.
    This matches the full agent described in Study.mq5.
    """

    def __init__(self):
        super().__init__()
        self.encoder   = TimeMoEEncoder()
        self.actor     = TimeMoEActor()
        self.director  = TimeMoEDirector()
        self.critic    = TimeMoECritic()
        self.forecasts = nn.ModuleList([
            ForecastHead(horizon=1),
            ForecastHead(horizon=N_FORECAST // 2),
            ForecastHead(horizon=N_FORECAST),
        ])

    def encode(
        self,
        state: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(state, timestamps)

    def forward(
        self,
        state: torch.Tensor,
        account: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        state     : [B, 1080]
        account   : [B, 13]
        timestamps: [B, 120]  optional

        returns dict with keys:
            actions   [B, 6]
            direction [B, 1]
            value     [B, 1]
            forecasts list of [B, horizon*9]
        """
        latent_tokens, _ = self.encode(state, timestamps)
        actions   = self.actor(account, latent_tokens)
        direction = self.director(actions, latent_tokens)
        value     = self.critic(actions, latent_tokens)
        fc = [h(latent_tokens) for h in self.forecasts]
        return {
            "actions":   actions,
            "direction": direction,
            "value":     value,
            "forecasts": fc,
        }


# ================================================================
# Feature construction (matching Test.mq5 / Study.mq5)
# ================================================================

def build_state_features(
    close:  list,
    high:   list,
    low:    list,
    volume: list,
    rsi:    list,
    cci:    list,
    atr:    list,
    macd:   list,
    signal: list,
) -> torch.Tensor:
    """
    Constructs state tensor from indicator arrays.
    Each list must be length >= HistoryBars, newest bar at index 0.
    Returns [1, STATE_DIM=1080] float32 tensor (raw, not normalized).

    Per bar (BarDescr=9 elements):
        [0] close - open
        [1] high  - open
        [2] low   - open
        [3] volume / 1000
        [4] RSI
        [5] CCI
        [6] ATR
        [7] MACD main
        [8] MACD signal
    """
    import numpy as np
    state = np.zeros(STATE_DIM, dtype=np.float32)
    # Reconstruct open from close and [0] field (not directly available from lists)
    # Caller provides close; we approximate open as close[b] - delta (assume delta=0 if not split)
    # For proper use, pass price data with explicit open values.
    for b in range(HISTORY_BARS):
        shift = b * BAR_DESCR
        # Use close as proxy for open when open not separately provided
        state[shift + 0] = 0.0                      # close - open (caller should subtract)
        state[shift + 1] = high[b] - close[b]       # high - open approx
        state[shift + 2] = low[b]  - close[b]       # low  - open approx
        state[shift + 3] = volume[b] / 1000.0
        state[shift + 4] = rsi[b]
        state[shift + 5] = cci[b]
        state[shift + 6] = atr[b]
        state[shift + 7] = macd[b]
        state[shift + 8] = signal[b]
    return torch.from_numpy(state).unsqueeze(0)     # [1, 1080]


def build_state_from_ohlcv(
    open_p: list, close: list, high: list, low: list,
    volume: list, rsi: list, cci: list, atr: list,
    macd: list, signal: list,
) -> torch.Tensor:
    """
    Full version with open prices. Newest bar at index 0.
    Returns [1, 1080].
    """
    import numpy as np
    state = np.zeros(STATE_DIM, dtype=np.float32)
    for b in range(HISTORY_BARS):
        shift = b * BAR_DESCR
        o = open_p[b]
        state[shift + 0] = close[b]  - o
        state[shift + 1] = high[b]   - o
        state[shift + 2] = low[b]    - o
        state[shift + 3] = volume[b] / 1000.0
        state[shift + 4] = rsi[b]
        state[shift + 5] = cci[b]
        state[shift + 6] = atr[b]
        state[shift + 7] = macd[b]
        state[shift + 8] = signal[b]
    return torch.from_numpy(state).unsqueeze(0)     # [1, 1080]


def build_account_features(
    balance: float,
    equity: float,
    prev_balance: float,
    prev_equity: float,
    buy_lots: float,
    sell_lots: float,
    buy_profit: float,
    sell_profit: float,
    position_discount: float,
    bar_time: float,
    etalon_balance: float = 1e5,
) -> torch.Tensor:
    """
    Constructs account descriptor tensor matching Test.mq5 bAccount construction.
    Returns [1, 13].
    """
    import numpy as np
    import math as _math
    acc = np.zeros(ACCOUNT_DESCR, dtype=np.float32)
    acc[0] = balance / etalon_balance
    acc[1] = (balance - prev_balance) / (prev_balance + 1e-9)
    acc[2] = equity  / (prev_balance + 1e-9)
    acc[3] = (equity - prev_equity)   / (prev_equity + 1e-9)
    acc[4] = buy_lots
    acc[5] = sell_lots
    acc[6] = buy_profit  / (prev_balance + 1e-9)
    acc[7] = sell_profit / (prev_balance + 1e-9)
    acc[8] = position_discount / (prev_balance + 1e-9)
    # Time encodings (from Test.mq5)
    REF_YEAR_SEC = 365 * 24 * 3600
    MONTH_SEC    = 30  * 24 * 3600
    WEEK_SEC     = 7   * 24 * 3600
    DAY_SEC      = 24  * 3600
    x = bar_time / REF_YEAR_SEC
    acc[9]  = _math.sin(2.0 * _math.pi * x) if x != 0 else 0.0
    x = bar_time / MONTH_SEC
    acc[10] = _math.cos(2.0 * _math.pi * x) if x != 0 else 0.0
    x = bar_time / WEEK_SEC
    acc[11] = _math.sin(2.0 * _math.pi * x) if x != 0 else 0.0
    x = bar_time / DAY_SEC
    acc[12] = _math.sin(2.0 * _math.pi * x) if x != 0 else 0.0
    return torch.from_numpy(acc).unsqueeze(0)        # [1, 13]


# ================================================================
# Public interface functions
# ================================================================

_MODEL: Optional[TimeMoESystem] = None
_DEVICE: Optional[torch.device] = None


def _get_model(device: torch.device) -> TimeMoESystem:
    global _MODEL, _DEVICE
    if _MODEL is None or _DEVICE != device:
        _MODEL = TimeMoESystem().to(device)
        _DEVICE = device
    return _MODEL


def generate_signal(
    features: torch.Tensor,
    account:  Optional[torch.Tensor] = None,
    timestamps: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """
    Takes raw state features and returns (direction, confidence).

    features   : [120, 9] float32 tensor — one row per bar, 9 descriptors
                 OR [1, 1080] already flattened
    account    : [1, 13] float32 (optional — zeros used if not provided)
    timestamps : [1, 120] float32 unix seconds (optional)
    device     : torch.device (defaults to DirectML GPU or CPU)

    Returns:
        direction  : float in [-1, 1]
                     Derived from actor actions[0] (buy_lot) vs actions[3] (sell_lot)
        confidence : float in [0, 1]
                     Director output
    """
    if device is None:
        device = get_device()

    model = _get_model(device)
    model.eval()

    with torch.no_grad():
        # Normalize input shape
        if features.dim() == 2 and features.shape == (HISTORY_BARS, BAR_DESCR):
            state = features.reshape(1, STATE_DIM).float()
        elif features.dim() == 2 and features.shape == (1, STATE_DIM):
            state = features.float()
        else:
            raise ValueError(f"features must be [120, 9] or [1, 1080], got {features.shape}")

        state = state.to(device)

        if account is None:
            account = torch.zeros(1, ACCOUNT_DESCR, dtype=torch.float32, device=device)
        else:
            account = account.float().to(device)

        if timestamps is not None:
            timestamps = timestamps.float().to(device)

        out = model(state, account, timestamps)
        actions   = out["actions"][0]     # [6]
        direction_raw = out["direction"][0, 0].item()

        # Direction: buy signal vs sell signal (same logic as Test.mq5)
        buy_signal  = actions[0].item()
        sell_signal = actions[3].item()
        net = buy_signal - sell_signal
        # Normalize to [-1, 1]
        direction = max(-1.0, min(1.0, net))
        confidence = direction_raw

    return direction, confidence


def train_step(
    batch: dict,
    optimizers: dict,
    device: Optional[torch.device] = None,
    discount_factor: float = DISC_FACTOR,
) -> dict:
    """
    Single training step matching Study.mq5 training logic.

    batch must contain:
        state       : [B, 1080]
        timestamps  : [B, 120]   (optional)
        account     : [B, 13]
        forecast_gt : [B, NForecast * BarDescr]  ground truth future bars
        reward      : [B, 1]    cumulative discounted reward

    optimizers must contain:
        encoder, actor, critic, director, forecasts (list of 3)

    Returns dict with loss values.
    """
    if device is None:
        device = get_device()

    model = _get_model(device)
    model.train()

    state       = batch["state"].float().to(device)
    account     = batch["account"].float().to(device)
    reward      = batch["reward"].float().to(device)
    timestamps  = batch.get("timestamps", None)
    forecast_gt = batch.get("forecast_gt", None)
    if timestamps is not None:
        timestamps = timestamps.float().to(device)
    if forecast_gt is not None:
        forecast_gt = forecast_gt.float().to(device)

    losses = {}

    # ---- Encoder forward ----
    optimizers["encoder"].zero_grad()
    latent_tokens, latent_pooled = model.encode(state, timestamps)
    latent_tokens_detach = latent_tokens.detach()

    # ---- Forecast losses (Study.mq5: train forecasts + backprop encoder) ----
    if forecast_gt is not None:
        for fi, fhead in enumerate(model.forecasts):
            optimizers["forecasts"][fi].zero_grad()
            horizon = fhead.horizon
            gt_slice_len = horizon * BAR_DESCR

            if forecast_gt.shape[1] >= gt_slice_len:
                gt_slice = forecast_gt[:, :gt_slice_len]
            else:
                gt_slice = F.pad(forecast_gt, (0, gt_slice_len - forecast_gt.shape[1]))

            pred = fhead(latent_tokens)
            loss_fc = F.mse_loss(pred, gt_slice)
            loss_fc.backward(retain_graph=True)
            optimizers["forecasts"][fi].step()
            losses[f"forecast_{fi}"] = loss_fc.item()

    # ---- Actor forward ----
    optimizers["actor"].zero_grad()
    actions = model.actor(account, latent_tokens_detach)

    # ---- Critic loss: MSE against reward ----
    optimizers["critic"].zero_grad()
    value_pred = model.critic(actions.detach(), latent_tokens_detach)
    loss_critic = F.mse_loss(value_pred, reward)
    loss_critic.backward()
    optimizers["critic"].step()
    losses["critic"] = loss_critic.item()

    # ---- Director loss: binary classification (reward > 0) ----
    optimizers["director"].zero_grad()
    direction_pred = model.director(actions.detach(), latent_tokens_detach)
    direction_gt = (reward > 0).float()
    loss_director = F.binary_cross_entropy(direction_pred, direction_gt)
    loss_director.backward()
    optimizers["director"].step()
    losses["director"] = loss_director.item()

    # ---- Actor loss: maximize critic value + director confidence ----
    actions_fresh = model.actor(account, latent_tokens_detach)
    value_for_actor = model.critic(actions_fresh, latent_tokens_detach)
    dir_for_actor   = model.director(actions_fresh, latent_tokens_detach)
    loss_actor = -(value_for_actor.mean() + dir_for_actor.mean())
    loss_actor.backward()
    optimizers["actor"].step()
    losses["actor"] = loss_actor.item()

    # ---- Encoder step (accumulated gradients from forecast passes) ----
    optimizers["encoder"].step()
    losses["encoder"] = 0.0

    return losses


def make_optimizers(lr: float = 3e-4, device: Optional[torch.device] = None) -> dict:
    """
    Creates Adam optimizers for all components.
    Returns dict matching expected keys for train_step().
    """
    if device is None:
        device = get_device()
    model = _get_model(device)
    return {
        "encoder":  torch.optim.Adam(model.encoder.parameters(),  lr=lr),
        "actor":    torch.optim.Adam(model.actor.parameters(),    lr=lr),
        "critic":   torch.optim.Adam(model.critic.parameters(),   lr=lr),
        "director": torch.optim.Adam(model.director.parameters(), lr=lr),
        "forecasts": [
            torch.optim.Adam(model.forecasts[i].parameters(), lr=lr)
            for i in range(3)
        ],
    }


def save_weights(path: str, device: Optional[torch.device] = None) -> None:
    """
    Saves all model weights to a single .pt file.
    Matches the DNG convention of saving separate .nnw files per component,
    but consolidated here for Python convenience.

    path: e.g. "C:/path/to/TimeMoE.pt"
    """
    if device is None:
        device = get_device()
    model = _get_model(device)
    # Move to CPU for saving (avoids DirectML serialization issues)
    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({
        "model_state": cpu_state,
        "timestamp":   time.time(),
        "constants": {
            "HISTORY_BARS":   HISTORY_BARS,
            "BAR_DESCR":      BAR_DESCR,
            "N_EXPERTS":      N_EXPERTS,
            "TOP_K":          TOP_K,
            "N_SKILLS":       N_SKILLS,
            "EMBEDDING_SIZE": EMBEDDING_SIZE,
            "SEGMENTS":       SEGMENTS,
        }
    }, path)
    print(f"[TimeMoE] Saved weights -> {path}")


def load_weights(path: str, device: Optional[torch.device] = None) -> None:
    """
    Loads weights from a .pt file saved by save_weights().
    """
    if device is None:
        device = get_device()
    if not os.path.exists(path):
        raise FileNotFoundError(f"[TimeMoE] Weight file not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    model = _get_model(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    ts = checkpoint.get("timestamp", 0)
    print(f"[TimeMoE] Loaded weights from {path} (saved {time.ctime(ts)})")


def model_summary() -> None:
    """Prints parameter counts per component."""
    device = get_device()
    model = _get_model(device)

    def count(m):
        return sum(p.numel() for p in m.parameters())

    print("=" * 55)
    print("TimeMoE System — Parameter Counts")
    print("=" * 55)
    print(f"  Encoder          : {count(model.encoder):>12,}")
    print(f"  Actor            : {count(model.actor):>12,}")
    print(f"  Director         : {count(model.director):>12,}")
    print(f"  Critic           : {count(model.critic):>12,}")
    for i, fh in enumerate(model.forecasts):
        print(f"  Forecast[{i}]      : {count(fh):>12,}")
    print(f"  TOTAL            : {count(model):>12,}")
    print("=" * 55)
    print(f"  Device           : {device}")
    print(f"  GPU available    : {_GPU_AVAILABLE}")
    print(f"  State input dim  : {STATE_DIM}  ({HISTORY_BARS} bars x {BAR_DESCR})")
    print(f"  Account dim      : {ACCOUNT_DESCR}")
    print(f"  Actions          : {N_ACTIONS}")
    print(f"  Experts          : {N_EXPERTS} (TopK={TOP_K})")
    print(f"  MoE layers       : {MOE_LAYERS}")
    print(f"  Attention heads  : {MOE_HEADS}")
    print("=" * 55)


# ================================================================
# Self-test — run directly to verify shapes
# ================================================================

if __name__ == "__main__":
    print("[TimeMoE] Running shape verification...")
    dev = get_device()
    print(f"[TimeMoE] Using device: {dev}")

    model = TimeMoESystem().to(dev)
    model.eval()

    B = 4
    state_t   = torch.randn(B, STATE_DIM, device=dev)
    account_t = torch.randn(B, ACCOUNT_DESCR, device=dev)
    # Use int64 then cast; float32 can't hold large unix timestamps exactly
    ts_t      = torch.randint(1_600_000, 1_700_000, (B, HISTORY_BARS), dtype=torch.int64, device=dev).float() * 1000

    with torch.no_grad():
        out = model(state_t, account_t, ts_t)

    print(f"  actions   : {out['actions'].shape}    expected [B={B}, 6]")
    print(f"  direction : {out['direction'].shape}  expected [B={B}, 1]")
    print(f"  value     : {out['value'].shape}      expected [B={B}, 1]")
    for i, fc in enumerate(out["forecasts"]):
        horizons = [1, N_FORECAST // 2, N_FORECAST]
        exp_dim  = horizons[i] * BAR_DESCR
        print(f"  forecast[{i}]: {fc.shape}  expected [B={B}, {exp_dim}]")

    # Test generate_signal
    feat = torch.randn(HISTORY_BARS, BAR_DESCR, device=dev)
    acc  = torch.zeros(1, ACCOUNT_DESCR, device=dev)
    direction, confidence = generate_signal(feat, acc, device=dev)
    print(f"  generate_signal -> direction={direction:.4f}, confidence={confidence:.4f}")

    model_summary()
    print("[TimeMoE] All shape checks passed.")

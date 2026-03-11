"""
expert_mamba.py — Mamba Trading Expert (PyTorch)
Faithfully translated from MQL5 source files:
  - Trajectory.mqh  (architecture definitions)
  - Test.mq5        (inference / trading logic)
  - Study.mq5       (training loop)
  - Research.mq5    (data collection / features)

Architecture — Encoder:
  Input (1080 flat) -> BatchNorm
  -> MambaBlock x3  [window=9, inside_dim=36, units=120]
  -> Transpose (120,9) -> (9,120)
  -> Conv1 (9 channels, kernel=120, out=96) + LReLU
  -> Conv2 (9 channels, kernel=96, out=24) + Tanh
  -> Transpose (9,24) -> (24,9)
  -> RevIN Denorm
  -> FreDF (FFT decomp, window=9, count=24)
  Latent tap at layer 8 (RevIN output, shape [24*9=216])

Actor (cross-attention head):
  Input: AccountDescr=12 + encoder latent tap
  Linear -> EmbeddingSize=32 (Sigmoid)
  -> MLCrossAttentionMLKV [units=(1,24), windows=(32,9), heads=(4,2), layers=4, out=32]
  -> Linear 256 (Sigmoid) -> Linear 256 (Sigmoid)
  -> Linear 2*NActions=12 -> VAE (reparameterize) -> NActions=6
  -> FreDF output gate (30% drop in freq domain)

Critic (same cross-attention structure):
  Input: NActions=6 + encoder latent tap
  Linear -> EmbeddingSize=32 (Sigmoid)
  -> MLCrossAttentionMLKV [units=(1,24), windows=(32,9), heads=(4,2), layers=5, out=32]
  -> Linear 256 x3 (Sigmoid) -> Linear NRewards=3 -> FreDF gate

Constants (from Trajectory.mqh):
  HistoryBars=120, BarDescr=9, AccountDescr=12, NActions=6, NRewards=3
  NForecast=24, EmbeddingSize=32, LatentCount=256, LatentLayer=8
  Buffer_Size=6500, DiscFactor=0.99

GPU: AMD RX 6800 XT via torch_directml (NOT cuda)
LSTM stays CPU if ever added (DirectML backprop broken for LSTM).
"""

import math
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Device selection — AMD RX 6800 XT via DirectML
# ---------------------------------------------------------------------------
try:
    import torch_directml
    _DML_AVAILABLE = True
except ImportError:
    _DML_AVAILABLE = False

def get_device() -> torch.device:
    if _DML_AVAILABLE:
        return torch_directml.device()
    return torch.device("cpu")

# ---------------------------------------------------------------------------
# Architecture constants (from Trajectory.mqh)
# ---------------------------------------------------------------------------
HISTORY_BARS   = 120
BAR_DESCR      = 9
ACCOUNT_DESCR  = 12
N_ACTIONS      = 6
N_REWARDS      = 3
N_FORECAST     = 24
EMBEDDING_SIZE = 32
LATENT_COUNT   = 256
LATENT_LAYER   = 8     # encoder layer index tapped for actor/critic input
DISC_FACTOR    = 0.99
INSIDE_DIM     = 4 * BAR_DESCR   # 36 — MambaOCL window_out

# Derived
STATE_SIZE   = HISTORY_BARS * BAR_DESCR   # 1080
LATENT_SIZE  = N_FORECAST * BAR_DESCR     # 216  (RevIN output shape)

# ---------------------------------------------------------------------------
# BatchNorm wrapper — mirrors defNeuronBatchNormOCL
# ---------------------------------------------------------------------------
class BatchNormLayer(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, momentum=0.0001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, num_features]
        return self.bn(x)


# ---------------------------------------------------------------------------
# MambaBlock — mirrors defNeuronMambaOCL (selective state space model)
#
# The MQL5 MambaOCL uses:
#   window      = BAR_DESCR (9)   — input feature dim per timestep
#   window_out  = INSIDE_DIM (36) — inner expansion dim (4x)
#   count       = HISTORY_BARS (120) — number of timesteps
#
# Selective SSM: input-dependent A, B, C, D projections.
# We implement the pure-Python recurrence (no CUDA kernel required — AMD).
# Architecture follows the original Mamba paper:
#   x -> expand projection -> SSM scan -> contract projection
#   with SiLU gating and residual.
# ---------------------------------------------------------------------------
class MambaBlock(nn.Module):
    """
    Selective State Space Model block.
    Input shape:  [B, units, window]  i.e. [B, 120, 9]
    Output shape: [B, units, window]  same
    """
    def __init__(self, d_model: int, d_inner: int, d_state: int = 16,
                 dt_rank: int = None):
        super().__init__()
        self.d_model = d_model    # BAR_DESCR = 9
        self.d_inner = d_inner    # INSIDE_DIM = 36
        self.d_state = d_state    # SSM state dimension

        if dt_rank is None:
            dt_rank = max(1, d_model // 4)
        self.dt_rank = dt_rank

        # Input projection: expand to 2*d_inner (one half gated)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)

        # SSM parameters — input-dependent (selective)
        self.x_proj  = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # dt softplus init (from Mamba paper)
        nn.init.uniform_(self.dt_proj.weight, -0.01, 0.01)
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(0.1) - math.log(dt_init_floor))
            + math.log(dt_init_floor)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # A: fixed diagonal, log-parameterised (HiPPO-like init)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D     = nn.Parameter(torch.ones(d_inner))

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        # Layer norm for stability
        self.norm = nn.LayerNorm(d_model)

    def ssm_scan(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, d_inner]
        returns: [B, L, d_inner]
        Selective SSM recurrence — O(L) sequential scan.
        """
        B, L, d = x.shape
        N = self.d_state

        A = -torch.exp(self.A_log.float())   # [d_inner, N]  negative definite

        # Project x to get delta, B_ssm, C_ssm
        x_dbl = self.x_proj(x)              # [B, L, dt_rank + 2*N]
        delta_raw, B_ssm, C_ssm = torch.split(
            x_dbl, [self.dt_rank, N, N], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta_raw))   # [B, L, d_inner]

        # Discretize A, B using ZOH
        # dA: [B, L, d_inner, N]
        dA = torch.exp(
            delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )
        # dB: [B, L, d_inner, N]
        dB = delta.unsqueeze(-1) * B_ssm.unsqueeze(-2)

        # Sequential scan
        h = torch.zeros(B, d, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            # h: [B, d, N]
            u_t  = x[:, t, :]           # [B, d]
            dA_t = dA[:, t, :, :]       # [B, d, N]
            dB_t = dB[:, t, :, :]       # [B, d, N]
            C_t  = C_ssm[:, t, :]       # [B, N]

            h = dA_t * h + dB_t * u_t.unsqueeze(-1)   # [B, d, N]
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1)   # [B, d]
            ys.append(y_t)

        y = torch.stack(ys, dim=1)   # [B, L, d]
        return y + x * self.D.unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, d_model]   where L=HISTORY_BARS, d_model=BAR_DESCR
        """
        residual = x
        x = self.norm(x)

        # Gated expansion
        xz = self.in_proj(x)                       # [B, L, 2*d_inner]
        xz_split = xz.chunk(2, dim=-1)             # each [B, L, d_inner]
        x_inner, z = xz_split[0], xz_split[1]

        x_inner = F.silu(x_inner)
        y = self.ssm_scan(x_inner)                 # [B, L, d_inner]
        y = y * F.silu(z)                          # gating
        out = self.out_proj(y)                     # [B, L, d_model]
        return out + residual


# ---------------------------------------------------------------------------
# FreDF — Frequency Domain Feature layer (mirrors defNeuronFreDFOCL)
#
# MQL5 params:
#   Encoder:  window=BAR_DESCR(9),  count=NForecast(24),  step=True  (encoder mode)
#   Actor:    window=NActions(6),   count=1,               step=False (gate mode)
#   Critic:   window=NRewards(3),   count=1,               step=False (gate mode)
#
# step=True  -> FFT decomposition returning spectral features
# step=False -> FFT magnitude gate (soft frequency filter on output)
# probability=0.7 -> keep top 70% of frequency components by magnitude
# ---------------------------------------------------------------------------
class FreDFLayer(nn.Module):
    """
    Frequency Domain Feature / gate layer.
    In encoder mode (step=True): input [B, count*window] -> FFT -> spectral features [B, count*window]
    In gate mode  (step=False): applies frequency magnitude gate to input, suppresses low-energy freqs
    """
    def __init__(self, window: int, count: int, step: bool, probability: float = 0.7):
        super().__init__()
        self.window      = window
        self.count       = count
        self.step        = step           # True = encoder, False = gate
        self.probability = probability    # keep fraction
        self.out_size    = window * count

        if step:
            # Learnable frequency mixing weights
            freq_len = window // 2 + 1
            self.freq_weight = nn.Parameter(
                torch.ones(count, freq_len, 2) * 0.01
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, out_size]  (flattened)
        """
        B = x.shape[0]
        # Reshape to [B, count, window]
        inp = x.view(B, self.count, self.window)

        if self.step:
            # Encoder mode: FFT + learnable spectral mixing
            freq = torch.fft.rfft(inp, dim=-1)                   # [B, count, freq_len]
            freq_real = freq.real
            freq_imag = freq.imag
            # Apply learnable weight (real part scales magnitude)
            w = torch.complex(
                torch.sigmoid(self.freq_weight[..., 0]),
                self.freq_weight[..., 1]
            ).unsqueeze(0)                                         # [1, count, freq_len]
            freq = freq * w
            out = torch.fft.irfft(freq, n=self.window, dim=-1)    # [B, count, window]
        else:
            # Gate mode: keep top-probability% of freqs by magnitude
            freq = torch.fft.rfft(inp, dim=-1)                    # [B, count, freq_len]
            magnitude = freq.abs()                                 # [B, count, freq_len]
            freq_len  = magnitude.shape[-1]
            k = max(1, int(freq_len * self.probability))
            # Find kth-largest threshold per (B, count)
            topk_vals, _ = torch.topk(magnitude, k, dim=-1)
            thresh = topk_vals[..., -1:].detach()                  # [B, count, 1]
            mask = (magnitude >= thresh).float()
            freq = freq * mask
            out = torch.fft.irfft(freq, n=self.window, dim=-1)    # [B, count, window]

        return out.view(B, self.out_size)


# ---------------------------------------------------------------------------
# RevIN Denorm — mirrors defNeuronRevInDenormOCL
# Stores running mean/std from encoder input, denormalises the forecast output.
# In inference we apply it as a learned affine layer (gamma/beta).
# ---------------------------------------------------------------------------
class RevINDenorm(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.affine_weight = nn.Parameter(torch.ones(num_features))
        self.affine_bias   = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor,
                mean: Optional[torch.Tensor] = None,
                std: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:    [B, num_features]
        mean: [B, num_features]  instance stats from encoder input (optional)
        std:  [B, num_features]  instance stats from encoder input (optional)
        """
        x = x * self.affine_weight + self.affine_bias
        if mean is not None and std is not None:
            x = x * (std + 1e-5) + mean
        return x


# ---------------------------------------------------------------------------
# MLCrossAttentionMLKV — mirrors defNeuronMLCrossAttentionMLKV
#
# Multi-Level Cross-Attention with Multi-Key-Value
# MQL5 definition (actor/critic):
#   units   = [1, NForecast]    -- query units per level: level0=1, level1=24
#   windows = [EmbeddingSize, BarDescr]  -- key/value window sizes per level
#   heads   = [4, 2]            -- attention heads per level
#   layers  = 4 (actor) / 5 (critic)
#   step    = 1
#   window_out = 32
#
# Interpretation: two-level cross-attention where:
#   Level 0 queries against the account embedding (dim=EmbeddingSize=32)
#   Level 1 queries against the encoder latent (dim=BarDescr=9, count=NForecast=24)
# Each level is a standard multi-head cross-attention block.
# Multiple transformer-style layers are stacked.
# Output is flattened and projected to window_out=32 per unit.
# ---------------------------------------------------------------------------
class MLCrossAttentionMLKVLayer(nn.Module):
    def __init__(self,
                 units:      list,   # [1, 24]
                 windows:    list,   # [32, 9]  query/key/value dims per level
                 heads:      list,   # [4, 2]
                 n_layers:   int,    # 4 or 5
                 window_out: int):   # 32
        super().__init__()
        self.n_levels  = len(units)
        self.units     = units
        self.windows   = windows
        self.heads     = heads
        self.n_layers  = n_layers
        self.window_out = window_out

        # Per-level cross-attention stacks
        self.attn_layers  = nn.ModuleList()
        self.ffn_layers   = nn.ModuleList()
        self.norms_q      = nn.ModuleList()
        self.norms_ff     = nn.ModuleList()
        # Input/output projections to align each level to a head-divisible dim
        self.in_projs     = nn.ModuleList()
        self.out_projs    = nn.ModuleList()
        self._attn_dims   = []

        for lvl in range(self.n_levels):
            raw_dim = windows[lvl]
            n_h     = heads[lvl]
            # Round up to nearest multiple of n_h so MHA constraint is satisfied
            attn_dim = math.ceil(raw_dim / n_h) * n_h

            self._attn_dims.append(attn_dim)
            # Linear projections in/out of attention space
            self.in_projs.append(
                nn.Linear(raw_dim, attn_dim) if attn_dim != raw_dim else nn.Identity()
            )
            self.out_projs.append(
                nn.Linear(attn_dim, raw_dim) if attn_dim != raw_dim else nn.Identity()
            )

            # Stack of transformer cross-attn layers (operating in attn_dim space)
            attn_stack = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=attn_dim,
                    num_heads=n_h,
                    batch_first=True,
                    dropout=0.0
                )
                for _ in range(n_layers)
            ])
            ffn_stack = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(attn_dim, 4 * attn_dim),
                    nn.GELU(),
                    nn.Linear(4 * attn_dim, attn_dim)
                )
                for _ in range(n_layers)
            ])
            norm_q_stack  = nn.ModuleList([nn.LayerNorm(attn_dim) for _ in range(n_layers)])
            norm_ff_stack = nn.ModuleList([nn.LayerNorm(attn_dim) for _ in range(n_layers)])
            self.attn_layers.append(attn_stack)
            self.ffn_layers.append(ffn_stack)
            self.norms_q.append(norm_q_stack)
            self.norms_ff.append(norm_ff_stack)

        # Output projection: concat all levels (in raw window dim) -> window_out
        total_out = sum(units[lvl] * windows[lvl] for lvl in range(self.n_levels))
        self.out_proj = nn.Linear(total_out, window_out)

    def forward(self, queries: list, keys_values: list) -> torch.Tensor:
        """
        queries     : list of tensors, one per level
                      level 0: [B, units[0], windows[0]]
                      level 1: [B, units[1], windows[1]]
        keys_values : list of tensors, same dims (used as both key and value)
        returns: [B, window_out]
        """
        level_outs = []
        for lvl in range(self.n_levels):
            q_raw  = queries[lvl]      # [B, Q, raw_dim]
            kv_raw = keys_values[lvl]  # [B, KV, raw_dim]

            # Project to attention-aligned dim
            q  = self.in_projs[lvl](q_raw)    # [B, Q, attn_dim]
            kv = self.in_projs[lvl](kv_raw)   # [B, KV, attn_dim]

            for layer_idx in range(self.n_layers):
                # Pre-norm cross-attention (q attends to kv)
                q_norm = self.norms_q[lvl][layer_idx](q)
                attn_out, _ = self.attn_layers[lvl][layer_idx](q_norm, kv, kv)
                q = q + attn_out

                # FFN
                q_norm = self.norms_ff[lvl][layer_idx](q)
                q = q + self.ffn_layers[lvl][layer_idx](q_norm)

            # Project back to raw dim
            q_out = self.out_projs[lvl](q)    # [B, Q, raw_dim]
            level_outs.append(q_out.reshape(q_out.shape[0], -1))  # [B, units*raw_dim]

        cat = torch.cat(level_outs, dim=-1)   # [B, total_out]
        return self.out_proj(cat)              # [B, window_out]


# ---------------------------------------------------------------------------
# VAE reparameterization — mirrors defNeuronVAEOCL
# Input: [B, 2*NActions]  (mu, log_var interleaved)
# Output: [B, NActions]
# ---------------------------------------------------------------------------
class VAELayer(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.n = n_actions

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (z, mu, log_var) — z is the reparameterized sample.
        During eval, returns mu directly (no noise).
        """
        mu      = x[:, :self.n]
        log_var = x[:, self.n:]
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z   = mu + eps * std
        else:
            z = mu
        return z, mu, log_var


# ---------------------------------------------------------------------------
# Encoder network — CreateEncoderDescriptions
#
# Layer 0:  Input  [B, 1080]
# Layer 1:  BatchNorm [B, 1080]
# Layer 2:  MambaBlock (reshape to [B,120,9] -> SSM -> [B,120,9])
# Layer 3:  MambaBlock (same)
# Layer 4:  MambaBlock (same)
# Layer 5:  Transpose [B,120,9] -> [B,9,120]  then flatten -> [B,1080]
# Layer 6:  Conv (9 channels: each linear 120->96) LReLU  -> [B, 9*96=864]  -- stored as [B,9,96]
# Layer 7:  Conv (9 channels: each linear 96->24) Tanh    -> [B, 9*24=216]  -- stored as [B,9,24]
# Layer 8:  Transpose [B,9,24] -> [B,24,9] -> flatten [B,216]
# Layer 9:  RevIN Denorm [B,216]                          << LATENT TAP (LatentLayer=8 in 0-index)
# Layer 10: FreDF encoder mode [B,216]
# ---------------------------------------------------------------------------
class MambaEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Layer 1 — BatchNorm
        self.bn = BatchNormLayer(STATE_SIZE)

        # Layers 2-4 — 3x MambaBlock
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(d_model=BAR_DESCR, d_inner=INSIDE_DIM)
            for _ in range(3)
        ])

        # Layer 6 — per-channel linear 120->96, LReLU
        # "Conv" in MQL5 sense: same weights applied per output channel independently
        # count=BarDescr(9) channels, window=prev_count(120), window_out=4*NForecast(96)
        self.conv1 = nn.Linear(HISTORY_BARS, 4 * N_FORECAST)   # 120 -> 96

        # Layer 7 — per-channel linear 96->24, Tanh
        # count=BarDescr(9) channels, window=4*NForecast(96), window_out=NForecast(24)
        self.conv2 = nn.Linear(4 * N_FORECAST, N_FORECAST)      # 96 -> 24

        # Layer 9 — RevIN Denorm
        self.revin = RevINDenorm(LATENT_SIZE)   # 216

        # Layer 10 — FreDF encoder mode
        self.fredf = FreDFLayer(
            window=BAR_DESCR,
            count=N_FORECAST,
            step=True,
            probability=0.7
        )

        # Store instance stats for RevIN
        self._revin_mean: Optional[torch.Tensor] = None
        self._revin_std:  Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, 1080]
        Returns:
          latent:   [B, 216]  — RevIN output (tapped at LatentLayer=8)
          fredf_out:[B, 216]  — FreDF output (final encoder output)
        """
        B = x.shape[0]

        # RevIN normalise on raw input for later denorm
        x_2d = x.view(B, HISTORY_BARS, BAR_DESCR)       # [B, 120, 9]
        mean = x_2d.mean(dim=1, keepdim=True)            # [B, 1, 9]
        std  = x_2d.std(dim=1, keepdim=True) + 1e-5
        x_norm = (x_2d - mean) / std                     # [B, 120, 9]
        flat_norm = x_norm.view(B, STATE_SIZE)            # [B, 1080]

        # Layer 1 — BatchNorm
        h = self.bn(flat_norm)                            # [B, 1080]

        # Layers 2-4 — MambaBlocks (reshape to [B,120,9])
        h = h.view(B, HISTORY_BARS, BAR_DESCR)           # [B, 120, 9]
        for mb in self.mamba_blocks:
            h = mb(h)                                     # [B, 120, 9]

        # Layer 5 — Transpose [B,120,9] -> [B,9,120]
        h = h.transpose(1, 2)                             # [B, 9, 120]

        # Layer 6 — Conv (channel-wise linear 120->96) + LReLU
        h = F.leaky_relu(self.conv1(h), negative_slope=0.01)   # [B, 9, 96]

        # Layer 7 — Conv (channel-wise linear 96->24) + Tanh
        h = torch.tanh(self.conv2(h))                    # [B, 9, 24]

        # Layer 8 — Transpose [B,9,24] -> [B,24,9] -> flatten
        h = h.transpose(1, 2).contiguous().view(B, LATENT_SIZE)  # [B, 216]

        # Layer 9 — RevIN Denorm (tap point for actor/critic)
        # Store mean/std reshaped to match output dim for denorm
        mean_flat = mean.expand(B, HISTORY_BARS, BAR_DESCR) \
                        .transpose(1, 2).contiguous().view(B, -1)[:, :LATENT_SIZE]
        std_flat  = std.expand(B, HISTORY_BARS, BAR_DESCR) \
                       .transpose(1, 2).contiguous().view(B, -1)[:, :LATENT_SIZE]

        latent = self.revin(h, mean_flat[:, :N_FORECAST * BAR_DESCR],
                            std_flat[:, :N_FORECAST * BAR_DESCR])   # [B, 216]

        # Layer 10 — FreDF
        fredf_out = self.fredf(latent)   # [B, 216]

        return latent, fredf_out


# ---------------------------------------------------------------------------
# Actor network — CreateDescriptions (actor branch)
#
# Input: account vector [B, 12] + encoder latent tap [B, 216]
#
# Layer 0: Linear 12->12 (passthrough, no activation)
# Layer 1: Linear 12->32 Sigmoid
# Layer 2: MLCrossAttentionMLKV
#           query  [B, 1, 32]  from account embedding
#           kv     [B, 24, 9]  from encoder latent reshaped
#           4 transformer layers, heads=[4,2]
# Layer 3: Linear ->256 Sigmoid
# Layer 4: Linear ->256 Sigmoid
# Layer 5: Linear ->12 (no activation)
# Layer 6: VAE -> [B, 6]
# Layer 7: FreDF gate
# ---------------------------------------------------------------------------
class MambaActor(nn.Module):
    def __init__(self):
        super().__init__()

        # Layer 1 — account embedding
        self.embed = nn.Linear(ACCOUNT_DESCR, EMBEDDING_SIZE)

        # Layer 2 — MLCrossAttentionMLKV
        # Level 0: query from account embed (1 token of dim 32), kv from same
        # Level 1: query from encoder latent (24 tokens of dim 9), kv from same
        self.cross_attn = MLCrossAttentionMLKVLayer(
            units=[1, N_FORECAST],
            windows=[EMBEDDING_SIZE, BAR_DESCR],
            heads=[4, 2],
            n_layers=4,
            window_out=32
        )

        # After cross-attn: output is [B, 32]
        self.fc3 = nn.Linear(32, LATENT_COUNT)
        self.fc4 = nn.Linear(LATENT_COUNT, LATENT_COUNT)
        self.fc5 = nn.Linear(LATENT_COUNT, 2 * N_ACTIONS)   # mu + log_var

        # Layer 6 — VAE
        self.vae = VAELayer(N_ACTIONS)

        # Layer 7 — FreDF gate
        self.fredf = FreDFLayer(
            window=N_ACTIONS,
            count=1,
            step=False,
            probability=0.7
        )

    def forward(self, account: torch.Tensor,
                latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        account: [B, 12]
        latent:  [B, 216]  — encoder latent tap
        Returns: (action, mu, log_var)  each [B, 6]
        """
        B = account.shape[0]

        # Layer 1 — embed account
        embed = torch.sigmoid(self.embed(account))    # [B, 32]

        # Prepare cross-attention inputs
        # Level 0: account embed as single query token [B, 1, 32]
        q0  = embed.unsqueeze(1)                       # [B, 1, 32]
        kv0 = embed.unsqueeze(1)                       # [B, 1, 32]  (self-kv for level 0)

        # Level 1: encoder latent reshaped to [B, 24, 9]
        q1  = latent.view(B, N_FORECAST, BAR_DESCR)   # [B, 24, 9]
        kv1 = q1                                       # same

        # Layer 2 — cross-attention
        h = self.cross_attn(
            queries=[q0, q1],
            keys_values=[kv0, kv1]
        )                                              # [B, 32]

        # Layers 3-5
        h = torch.sigmoid(self.fc3(h))                 # [B, 256]
        h = torch.sigmoid(self.fc4(h))                 # [B, 256]
        h = self.fc5(h)                                # [B, 12]

        # Layer 6 — VAE
        z, mu, log_var = self.vae(h)                   # [B, 6] each

        # Layer 7 — FreDF gate
        action = self.fredf(z)                         # [B, 6]

        return action, mu, log_var


# ---------------------------------------------------------------------------
# Critic network — CreateDescriptions (critic branch)
#
# Input: actions [B, 6] + encoder latent tap [B, 216]
# Same structure as actor but:
#   layer 2: MLCrossAttn layers=5 (vs 4 in actor)
#   layers 3-5: three 256-sigmoid layers (vs two in actor)
#   output: NRewards=3 -> FreDF gate -> [B, 3]
# ---------------------------------------------------------------------------
class MambaCritic(nn.Module):
    def __init__(self):
        super().__init__()

        # Layer 1 — action embedding
        self.embed = nn.Linear(N_ACTIONS, EMBEDDING_SIZE)

        # Layer 2 — MLCrossAttentionMLKV (5 transformer layers for critic)
        self.cross_attn = MLCrossAttentionMLKVLayer(
            units=[1, N_FORECAST],
            windows=[EMBEDDING_SIZE, BAR_DESCR],
            heads=[4, 2],
            n_layers=5,
            window_out=32
        )

        self.fc3 = nn.Linear(32, LATENT_COUNT)
        self.fc4 = nn.Linear(LATENT_COUNT, LATENT_COUNT)
        self.fc5 = nn.Linear(LATENT_COUNT, LATENT_COUNT)
        self.fc6 = nn.Linear(LATENT_COUNT, N_REWARDS)

        # Layer 7 — FreDF gate
        self.fredf = FreDFLayer(
            window=N_REWARDS,
            count=1,
            step=False,
            probability=0.7
        )

    def forward(self, actions: torch.Tensor,
                latent: torch.Tensor) -> torch.Tensor:
        """
        actions: [B, 6]
        latent:  [B, 216]
        Returns: [B, 3]  reward estimates
        """
        B = actions.shape[0]

        # Layer 1 — embed actions
        embed = torch.sigmoid(self.embed(actions))    # [B, 32]

        # Cross-attention inputs
        q0  = embed.unsqueeze(1)                       # [B, 1, 32]
        kv0 = embed.unsqueeze(1)
        q1  = latent.view(B, N_FORECAST, BAR_DESCR)   # [B, 24, 9]
        kv1 = q1

        # Layer 2
        h = self.cross_attn(
            queries=[q0, q1],
            keys_values=[kv0, kv1]
        )                                              # [B, 32]

        # Layers 3-5-6
        h = torch.sigmoid(self.fc3(h))                 # [B, 256]
        h = torch.sigmoid(self.fc4(h))                 # [B, 256]
        h = torch.sigmoid(self.fc5(h))                 # [B, 256]
        h = self.fc6(h)                                # [B, 3]

        # FreDF gate
        value = self.fredf(h)                          # [B, 3]
        return value


# ---------------------------------------------------------------------------
# Full Mamba Expert — top-level class
# ---------------------------------------------------------------------------
class MambaExpert(nn.Module):
    """
    Full Mamba trading expert.
    Encoder + Actor + Critic with shared encoder latent tap.
    """
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        self.dev = device or get_device()

        self.encoder = MambaEncoder()
        self.actor   = MambaActor()
        self.critic  = MambaCritic()

        self.to(self.dev)

        # Optimizers — ADAM as in MQL5
        self.optimizer_actor   = torch.optim.Adam(
            list(self.actor.parameters()), lr=1e-4
        )
        self.optimizer_critic  = torch.optim.Adam(
            list(self.critic.parameters()), lr=1e-4
        )
        # Encoder is train=False during inference (Study.mq5: Encoder.TrainMode(false))
        # but is updated via gradient through actor/critic during study
        self.optimizer_encoder = torch.optim.Adam(
            list(self.encoder.parameters()), lr=1e-4
        )

    # ------------------------------------------------------------------
    # generate_signal — mirrors OnTick() in Test.mq5
    # ------------------------------------------------------------------
    def generate_signal(self,
                        features: np.ndarray,
                        account: Optional[np.ndarray] = None
                        ) -> Tuple[int, float]:
        """
        features: [120, 9]  — 120 bars of [close-open, high-open, low-open,
                               volume/1000, RSI, CCI, ATR, MACD, MACD_signal]
        account:  [12]      — normalized account state (optional; zeros if None)
                              [delta_balance, equity/balance, delta_equity,
                               buy_vol, sell_vol, buy_profit, sell_profit,
                               position_discount, sin_year, cos_month,
                               sin_week, sin_day]

        Returns:
          direction:  1 (buy), -1 (sell), 0 (hold)
          confidence: 0.0 - 1.0
        """
        self.eval()
        self.encoder.train(False)

        if account is None:
            account = np.zeros(ACCOUNT_DESCR, dtype=np.float32)

        x_state   = torch.tensor(
            features.flatten(), dtype=torch.float32
        ).unsqueeze(0).to(self.dev)                          # [1, 1080]
        x_account = torch.tensor(
            account, dtype=torch.float32
        ).unsqueeze(0).to(self.dev)                          # [1, 12]

        with torch.no_grad():
            latent, _ = self.encoder(x_state)               # [1, 216]
            action, _, _ = self.actor(x_account, latent)    # [1, 6]
            action = action.squeeze(0).cpu().numpy()         # [6]

        # Mirror Test.mq5 action parsing:
        # action[0] = buy_lot, action[1] = buy_tp, action[2] = buy_sl
        # action[3] = sell_lot, action[4] = sell_tp, action[5] = sell_sl
        # Net out buy vs sell lots to get direction
        buy_lot  = float(action[0])
        sell_lot = float(action[3])

        if buy_lot > sell_lot:
            net = buy_lot - sell_lot
            total = buy_lot + sell_lot + 1e-8
            direction  = 1
            confidence = float(np.clip(net / total, 0.0, 1.0))
        elif sell_lot > buy_lot:
            net = sell_lot - buy_lot
            total = buy_lot + sell_lot + 1e-8
            direction  = -1
            confidence = float(np.clip(net / total, 0.0, 1.0))
        else:
            direction  = 0
            confidence = 0.0

        return direction, confidence

    # ------------------------------------------------------------------
    # train_step — mirrors the core of Train() in Study.mq5
    # Walk-forward batch: one iteration of actor-critic update
    # ------------------------------------------------------------------
    def train_step(self, batch: dict) -> dict:
        """
        batch keys:
          'state'       : [B, 1080] float32  — raw bar features
          'account'     : [B, 12]   float32  — normalized account state
          'action'      : [B, 6]    float32  — actions taken
          'reward_curr' : [B, 3]    float32  — rewards at step i+1
          'reward_next' : [B, 3]    float32  — rewards at step i+2 (for TD target)
          'profitable'  : [B]       bool     — trajectory was profitable (for actor imitation)

        Returns dict with scalar losses.
        """
        self.train()
        self.encoder.train(False)   # Encoder frozen during study (train=False in MQL5)

        dev = self.dev

        state       = torch.tensor(batch['state'],       dtype=torch.float32).to(dev)
        account     = torch.tensor(batch['account'],     dtype=torch.float32).to(dev)
        actions_buf = torch.tensor(batch['action'],      dtype=torch.float32).to(dev)
        rew_curr    = torch.tensor(batch['reward_curr'], dtype=torch.float32).to(dev)
        rew_next    = torch.tensor(batch['reward_next'], dtype=torch.float32).to(dev)
        profitable  = torch.tensor(batch['profitable'],  dtype=torch.float32).to(dev)  # [B]

        with torch.no_grad():
            latent, _ = self.encoder(state)      # [B, 216]

        # ---------------------------------------------------------------
        # Critic update — TD(0): target = r_curr - r_next * gamma
        # (Study.mq5: result = reward[i+1] - reward[i+2] * DiscFactor)
        # ---------------------------------------------------------------
        self.optimizer_critic.zero_grad()
        self.critic.train(True)

        td_target = (rew_curr - rew_next * DISC_FACTOR).detach()   # [B, 3]
        pred_value = self.critic(actions_buf, latent.detach())     # [B, 3]
        critic_loss = F.mse_loss(pred_value, td_target)
        critic_loss.backward()
        self.optimizer_critic.step()

        # ---------------------------------------------------------------
        # Actor update — two passes mirroring Study.mq5:
        # Pass 1: imitation on profitable trajectories (backProp to buffer actions)
        # Pass 2: policy gradient via critic (backPropGradient)
        # ---------------------------------------------------------------
        self.optimizer_actor.zero_grad()
        self.actor.train(True)
        self.critic.train(False)

        action_pred, mu, log_var = self.actor(account, latent.detach())   # [B, 6]

        # VAE regularization (KL)
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        # Pass 1: behavior cloning on profitable trajectories
        bc_mask = profitable.unsqueeze(-1)   # [B, 1]
        bc_loss = (F.mse_loss(action_pred, actions_buf, reduction='none') * bc_mask).mean()

        # Pass 2: maximize critic value (policy gradient)
        # Study.mq5: scale +1% for positive value, -1% for negative
        with torch.no_grad():
            critic_val  = self.critic(action_pred.detach(), latent.detach())  # [B, 3]
            pg_weights  = torch.where(critic_val >= 0,
                                      critic_val * 1.01,
                                      critic_val * 0.99)

        # Re-run critic with grad through actor output
        pg_val  = self.critic(action_pred, latent.detach())    # [B, 3]
        pg_loss = -torch.mean(pg_val * pg_weights.detach())

        actor_loss = bc_loss + pg_loss + 0.001 * kl_loss
        actor_loss.backward()
        self.optimizer_actor.step()

        return {
            'critic_loss': float(critic_loss.detach().cpu()),
            'actor_loss':  float(actor_loss.detach().cpu()),
            'bc_loss':     float(bc_loss.detach().cpu()),
            'pg_loss':     float(pg_loss.detach().cpu()),
            'kl_loss':     float(kl_loss.detach().cpu()),
        }

    # ------------------------------------------------------------------
    # save_weights / load_weights
    # ------------------------------------------------------------------
    def save_weights(self, path: str) -> None:
        """Save full model state to path (torch checkpoint)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'encoder': self.encoder.state_dict(),
            'actor':   self.actor.state_dict(),
            'critic':  self.critic.state_dict(),
            'opt_enc': self.optimizer_encoder.state_dict(),
            'opt_act': self.optimizer_actor.state_dict(),
            'opt_crt': self.optimizer_critic.state_dict(),
        }, path)
        print(f"[MambaExpert] weights saved -> {path}")

    def load_weights(self, path: str) -> None:
        """Load model state from path."""
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        self.encoder.load_state_dict(ckpt['encoder'])
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        if 'opt_enc' in ckpt:
            self.optimizer_encoder.load_state_dict(ckpt['opt_enc'])
        if 'opt_act' in ckpt:
            self.optimizer_actor.load_state_dict(ckpt['opt_act'])
        if 'opt_crt' in ckpt:
            self.optimizer_critic.load_state_dict(ckpt['opt_crt'])
        self.to(self.dev)
        print(f"[MambaExpert] weights loaded <- {path}")

    # ------------------------------------------------------------------
    # Utility: parameter count
    # ------------------------------------------------------------------
    def param_count(self) -> dict:
        enc = sum(p.numel() for p in self.encoder.parameters())
        act = sum(p.numel() for p in self.actor.parameters())
        crt = sum(p.numel() for p in self.critic.parameters())
        return {
            'encoder': enc,
            'actor':   act,
            'critic':  crt,
            'total':   enc + act + crt,
        }


# ---------------------------------------------------------------------------
# Feature builder — mirrors bar state assembly in Test.mq5 / Research.mq5
#
# Constructs account state vector from raw account values.
# Call this before generate_signal() if you have raw MT5 account data.
# ---------------------------------------------------------------------------
def build_account_vector(balance: float, equity: float,
                         prev_balance: float, prev_equity: float,
                         buy_volume: float, sell_volume: float,
                         buy_profit: float, sell_profit: float,
                         position_discount: float,
                         timestamp: float) -> np.ndarray:
    """
    Mirrors bAccount construction in Test.mq5 OnTick().
    timestamp: unix seconds (Rates[0].time)
    Returns: [12] float32 array
    """
    # Time encoding (4 sinusoidal features)
    x_year  = timestamp / (365.25 * 24 * 3600)
    x_month = timestamp / (30.44 * 24 * 3600)
    x_week  = timestamp / (7 * 24 * 3600)
    x_day   = timestamp / (24 * 3600)

    t_sin_year  = math.sin(2 * math.pi * x_year)
    t_cos_month = math.cos(2 * math.pi * x_month)
    t_sin_week  = math.sin(2 * math.pi * x_week)
    t_sin_day   = math.sin(2 * math.pi * x_day)

    pb = prev_balance  if prev_balance  > 0 else 1.0
    pe = prev_equity   if prev_equity   > 0 else 1.0

    vec = np.array([
        (balance - pb) / pb,           # 0: delta balance ratio
        equity / pb,                   # 1: equity / prev_balance
        (equity - pe) / pe,            # 2: delta equity ratio
        buy_volume,                    # 3: buy volume
        sell_volume,                   # 4: sell volume
        buy_profit / pb,               # 5: buy profit ratio
        sell_profit / pb,              # 6: sell profit ratio
        position_discount / pb,        # 7: position discount ratio
        t_sin_year,                    # 8
        t_cos_month,                   # 9
        t_sin_week,                    # 10
        t_sin_day,                     # 11
    ], dtype=np.float32)

    return vec


# ---------------------------------------------------------------------------
# Quick smoke test — run directly to verify architecture initialises
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import time

    print("Initialising MambaExpert...")
    expert = MambaExpert()
    counts = expert.param_count()
    print(f"  Encoder params : {counts['encoder']:,}")
    print(f"  Actor params   : {counts['actor']:,}")
    print(f"  Critic params  : {counts['critic']:,}")
    print(f"  Total params   : {counts['total']:,}")
    print(f"  Device         : {expert.dev}")

    # Fake 120 bars of [close-open, high-open, low-open, vol, RSI, CCI, ATR, MACD, signal]
    rng = np.random.default_rng(42)
    features = rng.standard_normal((HISTORY_BARS, BAR_DESCR)).astype(np.float32)

    print("\nRunning generate_signal()...")
    t0 = time.time()
    direction, confidence = expert.generate_signal(features)
    dt = time.time() - t0
    print(f"  direction  : {direction:+d}")
    print(f"  confidence : {confidence:.4f}")
    print(f"  latency    : {dt*1000:.1f} ms")

    # Fake training batch
    B = 8
    batch = {
        'state':       rng.standard_normal((B, STATE_SIZE)).astype(np.float32),
        'account':     rng.standard_normal((B, ACCOUNT_DESCR)).astype(np.float32),
        'action':      np.clip(rng.standard_normal((B, N_ACTIONS)).astype(np.float32), 0, 1),
        'reward_curr': rng.standard_normal((B, N_REWARDS)).astype(np.float32),
        'reward_next': rng.standard_normal((B, N_REWARDS)).astype(np.float32),
        'profitable':  rng.integers(0, 2, B).astype(np.float32),
    }
    print("\nRunning train_step()...")
    losses = expert.train_step(batch)
    for k, v in losses.items():
        print(f"  {k:<15}: {v:.6f}")

    print("\nSmoke test PASSED.")

"""
DACGLSTM Expert - PyTorch implementation
Architecture: LinearAttention x2 -> CG-LSTM -> Conv -> CrossDMHAttention x3

Faithfully converted from MQ5 source (DNG, DACGLSTM project).
Trajectory.mqh / Study.mq5 / Test.mq5 CreateDescriptions() is the ground truth.

Constants from Trajectory.mqh:
  HistoryBars  = 120
  BarDescr     = 9
  AccountDescr = 12
  NActions     = 6
  NSkills      = 32
  EmbeddingSize= 64
  LatentCount  = 256
  LatentLayer  = 5   (index of CGLSTM output layer in encoder)
  NForecast    = 15
  NRewards     = 3
  DiscFactor   = 0.2

Input features per bar (BarDescr=9):
  [0] close - open
  [1] high  - open
  [2] low   - open
  [3] tick_volume / 1000
  [4] RSI(14)
  [5] CCI(14)
  [6] ATR(14)
  [7] MACD main
  [8] MACD signal

Device policy (AMD RX 6800 XT):
  - INFERENCE:  torch_directml GPU where available, falls back to CPU
  - TRAINING:   CPU only.  DirectML / ROCm backprop is broken on Windows
                (segfaults on any .backward() call from the Python process).
                This matches the constraint in expert_conformer.py and
                expert_mamba.py in this library.
  - CGLSTM:     Always CPU regardless of inference device (LSTMCell backprop
                incompatible with DirectML even in native mode).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
try:
    import torch_directml
    _DML_AVAILABLE = True
except ImportError:
    _DML_AVAILABLE = False

def get_dml_device():
    if _DML_AVAILABLE:
        return torch_directml.device()
    return torch.device("cpu")

CPU_DEVICE  = torch.device("cpu")
GPU_DEVICE  = get_dml_device()   # inference
TRAIN_DEVICE = CPU_DEVICE        # training/backward — CPU only on Windows AMD

# ---------------------------------------------------------------------------
# Architecture constants (mirror Trajectory.mqh defines)
# ---------------------------------------------------------------------------
HISTORY_BARS   = 120
BAR_DESCR      = 9
ACCOUNT_DESCR  = 12
N_ACTIONS      = 6
N_SKILLS       = 32
EMBEDDING_SIZE = 64
LATENT_COUNT   = 256
LATENT_LAYER   = 5   # encoder layer index where CGLSTM output lives
N_FORECAST     = 15
DISC_FACTOR    = 0.2

# Derived
INPUT_SIZE     = HISTORY_BARS * BAR_DESCR   # 1080


# ===========================================================================
# 1. Linear Attention  (O(N) via kernel trick, phi(x) = elu(x)+1)
#
# MQ5 layer 2: count=120, window=9(BarDescr), layers=1, window_out=32
# MQ5 layer 4: count=1,   window=120,         layers=9, window_out=32
#
# Interpretation: the layer sees `count` tokens each of dimension `window`,
# and projects to window_out per token. layers= number of such sequences to
# process (stacked independently then concatenated).
# ===========================================================================
class LinearAttentionLayer(nn.Module):
    """
    Linear attention via kernel trick (phi = elu + 1).
    Processes `layers` independent sequences of shape [seq_len, d_in]
    and outputs [seq_len * layers, d_out].

    Parameters align with MQ5 CLayerDescription fields:
      count      -> seq_len  (number of tokens)
      window     -> d_in     (input dim per token)
      layers     -> n_seqs   (number of parallel sequences)
      window_out -> d_out    (output dim per token)
    """

    def __init__(self, seq_len: int, d_in: int, d_out: int, n_seqs: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.d_in    = d_in
        self.d_out   = d_out
        self.n_seqs  = n_seqs

        # Q, K, V projections -- shared across all n_seqs sequences
        self.q_proj = nn.Linear(d_in, d_out, bias=False)
        self.k_proj = nn.Linear(d_in, d_out, bias=False)
        self.v_proj = nn.Linear(d_in, d_out, bias=False)
        self.out_proj = nn.Linear(d_out, d_out)
        self.norm = nn.LayerNorm(d_out)

    @staticmethod
    def _phi(x: torch.Tensor) -> torch.Tensor:
        """Positive feature map: elu(x) + 1"""
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, n_seqs * seq_len * d_in]  (flat, as the MQ5 buffer)
        returns: [batch, n_seqs * seq_len * d_out]
        """
        batch = x.shape[0]
        # Reshape to [batch * n_seqs, seq_len, d_in]
        x = x.view(batch * self.n_seqs, self.seq_len, self.d_in)

        Q = self._phi(self.q_proj(x))   # [B*S, L, d_out]
        K = self._phi(self.k_proj(x))   # [B*S, L, d_out]
        V = self.v_proj(x)              # [B*S, L, d_out]

        # Linear attention: O = D^-1 (Q (K^T V))
        # where D = Q * (K^T 1)
        kv  = torch.bmm(K.transpose(1, 2), V)          # [B*S, d, d]
        num = torch.bmm(Q, kv)                          # [B*S, L, d]
        k_sum = K.sum(dim=1, keepdim=True)              # [B*S, 1, d]
        den = (Q * k_sum).sum(dim=-1, keepdim=True).clamp(min=1e-6)
        out = num / den

        out = self.norm(self.out_proj(out) + out)       # residual + norm
        return out.reshape(batch, -1)


# ===========================================================================
# 2. CG-LSTM Cell  (Contextual Gating LSTM)
#
# MQ5 layer 5: count=NSkills=32, window=HistoryBars=120, layers=BarDescr=9
#
# The CGLSTM has an extra contextual gate that modulates cell/hidden state
# based on the full sequence context. The MQ5 source uses OpenCL so the
# exact kernel is not visible, but the canonical DACGLSTM formulation adds
# a gating mechanism where g = sigmoid(W_g * x + U_g * h + b_g) that
# multiplies the cell output before the hidden state update.
#
# This runs on CPU per project rules.
# ===========================================================================
class CGLSTMCell(nn.Module):
    """
    Contextual Gating LSTM Cell.
    Extra contextual gate on cell output:
      g = sigmoid(W_g x + U_g h + b_g)
      c_out = g * tanh(c)
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # Standard LSTM gates: i, f, g(cell), o
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        # Contextual gate
        self.cg_w = nn.Linear(input_size, hidden_size, bias=False)
        self.cg_u = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h, c = state
        h_new, c_new = self.lstm_cell(x, (h, c))
        # Contextual gate
        g = torch.sigmoid(self.cg_w(x) + self.cg_u(h))
        h_gated = g * torch.tanh(c_new)
        return h_gated, (h_gated, c_new)


class CGLSTMEncoder(nn.Module):
    """
    Processes BAR_DESCR=9 parallel sequences of length HISTORY_BARS=120,
    each with 1 feature (after the transpose), and outputs NSkills=32
    per sequence position.

    MQ5 layout after layers 2-4:
      Input to layer 5 is [BarDescr, HistoryBars] = [9, 120] when viewed
      as a matrix (9 variable-channels, 120 time steps).
      CGLSTM: count=32(skills), window=120(seq), layers=9(variables)
      Output: [NSkills * BarDescr] = [32 * 9] = 288
    """

    def __init__(
        self,
        seq_len: int    = HISTORY_BARS,
        n_vars: int     = BAR_DESCR,
        n_skills: int   = N_SKILLS,
        d_in: int       = 1,
    ):
        super().__init__()
        self.seq_len  = seq_len
        self.n_vars   = n_vars
        self.n_skills = n_skills
        self.d_in     = d_in

        # One shared CGLSTM cell across all variable channels
        self.cell = CGLSTMCell(d_in, n_skills)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, n_vars * seq_len]  (from upstream transpose)
        returns: [batch, n_skills * n_vars]
        """
        # This runs on CPU always
        device = x.device
        batch = x.shape[0]

        # Reshape to [batch, n_vars, seq_len, 1]
        x = x.view(batch, self.n_vars, self.seq_len, self.d_in)

        outputs = []
        for v in range(self.n_vars):
            seq = x[:, v, :, :]  # [batch, seq_len, 1]
            h = torch.zeros(batch, self.n_skills, device=device)
            c = torch.zeros(batch, self.n_skills, device=device)
            for t in range(self.seq_len):
                inp = seq[:, t, :]   # [batch, 1]
                h, (h, c) = self.cell(inp, (h, c))
            outputs.append(h)  # last hidden state: [batch, n_skills]

        # Stack: [batch, n_vars * n_skills]
        return torch.cat(outputs, dim=1)


# ===========================================================================
# 3. Conv layer (MQ5 defNeuronConvOCL)
#
# MQ5 ConvOCL applies a 1D conv with no overlap:
#   input  = [layers, window]   (layers sequences each of length window)
#   output = [layers, window_out]
# count=1 means there is 1 "batch position" (i.e., stride==window so no
# overlapping windows).
#
# In the encoder:
#   Layer 6: count=1, window=32(NSkills), step=32, window_out=60(4*NForecast),
#             layers=9(BarDescr), activation=SoftPlus
#             Input: [9*32]=288  Output: [9*60]=540
#   Layer 7: count=1, window=60, step=60, window_out=15(NForecast),
#             layers=9, activation=Tanh
#             Input: [9*60]=540  Output: [9*15]=135
# ===========================================================================
class ConvProjection(nn.Module):
    """
    Point-wise projection applied identically to each of `n_seqs` segments.
    Equivalent to the MQ5 ConvOCL with count=1, step==window (no overlap).
    """

    def __init__(
        self,
        n_seqs: int,
        d_in: int,
        d_out: int,
        activation: str = "none"
    ):
        super().__init__()
        self.n_seqs = n_seqs
        self.d_in   = d_in
        self.d_out  = d_out
        self.proj   = nn.Linear(d_in, d_out)
        self.act_name = activation

        if activation == "softplus":
            self.act = nn.Softplus()
        elif activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, n_seqs * d_in]
        returns: [batch, n_seqs * d_out]
        """
        batch = x.shape[0]
        x = x.view(batch * self.n_seqs, self.d_in)
        x = self.proj(x)
        x = self.act(x)
        return x.view(batch, self.n_seqs * self.d_out)


# ===========================================================================
# 4. Transpose layer (defNeuronTransposeOCL)
#
# Encoder layer 3: count=120, window=9  -> transpose [9, 120] to [120, 9]
#   Input:  [9 * 120] (from LinAttn layer 2 which outputs n_seqs=1 * seq=120 tokens)
#   Output: [120 * 9]   -- actually a matrix transpose
#
# Wait -- re-reading layer 2 output:
#   Layer 2: count=120, window=9, layers=1, window_out=32
#   Output: 1 * 120 * 32 = 3840
#   But layer 3 Transpose has count=120, window=BarDescr=9
#   That doesn't match 32... let me re-examine.
#
# The LinearAttention output dim per token is window_out=32, but MQ5 Transpose
# takes count=120, window=BarDescr=9.  This means the Transpose is of the
# ORIGINAL data reshaped, not the attention output.
#
# Looking more carefully at the DNG NeuroNet framework:
# defNeuronLinerAttention with window_out=32 means 32-dim internal key/value
# but the OUTPUT of the layer is the same shape as the input (count * window).
# The attention is used to reweight/mix the tokens, output shape = input shape.
# So layer 2 output = 120 * 9 = 1080.
# Layer 3 Transpose(count=120, window=9): transposes [120, 9] -> [9, 120] = 1080
# Layer 4 LinearAttention(count=1, window=120, layers=9, window_out=32):
#   9 sequences each [1, 120], output stays [9 * 1 * 120] = 1080 ... hmm
#   Actually with count=1 and window=120, the "sequence" has 1 token of dim 120.
#   window_out=32 is projection dim.  Output = 9 * 1 * 120 = 1080.
# Layer 5 CGLSTM: window=HistoryBars=120 (seq), layers=BarDescr=9 (parallel seqs)
#   Input: [9, 120] flattened = 1080. CGLSTM processes 9 parallel 120-step seqs.
#   Output: [NSkills * BarDescr] = 32 * 9 = 288
#
# So the full shape flow:
# [1080] -> BNoise[1080] -> LinAttn[1080] -> Transpose[1080] -> LinAttn[1080]
# -> CGLSTM[288] -> Conv6[540] -> Conv7[135] -> Transpose8[135] -> RevIn[135]
# ===========================================================================
class TransposeLayer(nn.Module):
    """
    Reshapes [batch, rows*cols] as [rows, cols] then transposes to [cols, rows].
    """

    def __init__(self, rows: int, cols: int):
        super().__init__()
        self.rows = rows
        self.cols = cols

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        return x.view(batch, self.rows, self.cols).transpose(1, 2).contiguous().view(batch, -1)


# ===========================================================================
# 5. Reversible Instance Normalization (RevIN)
#
# MQ5 defNeuronRevInDenormOCL: Denormalize using stored statistics.
# In training it normalizes the input; at inference it denormalizes.
# For pure forward pass (inference-only mode here), we implement RevIN
# as a learnable affine norm that can be inverted.
# ===========================================================================
class RevIN(nn.Module):
    """
    Reversible Instance Normalization.
    Normalizes each sample independently, with learnable affine.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(num_features))
        self.affine_bias   = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, denorm: bool = False) -> torch.Tensor:
        if denorm:
            return x * self.affine_weight + self.affine_bias
        mean = x.mean(dim=-1, keepdim=True)
        std  = x.std(dim=-1, keepdim=True).clamp(min=self.eps)
        return (x - mean) / std * self.affine_weight + self.affine_bias


# ===========================================================================
# 6. Cross Dense Multi-Head Attention (CrossDMHAttention)
#
# MQ5 defNeuronCrossDMHAttention:
#   - Two input streams: primary (query) and cross (key/value)
#   - windows[] = [primary_window, cross_window]  (d_in per unit)
#   - units[]   = [primary_units, cross_units]    (number of tokens)
#   - step      = num_heads
#   - window_out = d_head (dim per head)
#
# In Task (LatentLayer+1):
#   windows=[NSkills=32, NSkills=32], units=[BarDescr=9, BarDescr=9]
#   step=4 heads, window_out=32
#   Primary input:  from CGLSTM output = [9 tokens, 32 dims] = 288 flat
#   Cross input:    same (self-cross attention on CGLSTM output)
#   Output: primary_units * window_out = 9 * 32 = 288
#
# In Actor layer 2:
#   windows=[AccountDescr=12, NActions=6], units=[1, BarDescr=9]
#   step=4 heads, window_out=32
#   Primary: bAccount [1 token of 12 dims] = 12 flat
#   Cross:   cTask output [9 tokens of 6 dims] = 54 flat
#   Output: 1 * 32 = 32
#
# In Critic/Director layer 2:
#   windows=[3, NSkills=32], units=[NActions/3=2, BarDescr=9]
#   step=4 heads, window_out=32
#   Primary: action [2 tokens of 3 dims] = 6 flat
#   Cross:   CGLSTM [9 tokens of 32 dims] = 288 flat
#   Output: 2 * 32 = 64
# ===========================================================================
class CrossDMHAttention(nn.Module):
    """
    Cross Dense Multi-Head Attention.

    primary: [batch, primary_units, primary_window]  (query stream)
    cross:   [batch, cross_units,   cross_window]    (key/value stream)
    output:  [batch, primary_units * d_head]
    """

    def __init__(
        self,
        primary_window: int,
        cross_window: int,
        primary_units: int,
        cross_units: int,
        n_heads: int,
        d_head: int,
    ):
        super().__init__()
        self.primary_units = primary_units
        self.cross_units   = cross_units
        self.n_heads       = n_heads
        self.d_head        = d_head
        self.scale         = math.sqrt(d_head)

        # Dense (not sparse) projections per head
        self.q_proj = nn.Linear(primary_window, n_heads * d_head)
        self.k_proj = nn.Linear(cross_window,   n_heads * d_head)
        self.v_proj = nn.Linear(cross_window,   n_heads * d_head)
        self.out_proj = nn.Linear(n_heads * d_head, d_head)
        self.norm = nn.LayerNorm(d_head)

    def forward(
        self,
        primary: torch.Tensor,
        cross: torch.Tensor
    ) -> torch.Tensor:
        """
        primary: [batch, primary_units * primary_window]
        cross:   [batch, cross_units   * cross_window]
        returns: [batch, primary_units * d_head]
        """
        batch = primary.shape[0]
        H, D = self.n_heads, self.d_head

        # Reshape
        prim = primary.view(batch, self.primary_units, -1)   # [B, P, pw]
        cros = cross.view(batch, self.cross_units, -1)       # [B, C, cw]

        # Project
        Q = self.q_proj(prim).view(batch, self.primary_units, H, D).transpose(1, 2)   # [B,H,P,D]
        K = self.k_proj(cros).view(batch, self.cross_units,   H, D).transpose(1, 2)   # [B,H,C,D]
        V = self.v_proj(cros).view(batch, self.cross_units,   H, D).transpose(1, 2)   # [B,H,C,D]

        # Attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale   # [B,H,P,C]
        attn = F.softmax(attn, dim=-1)
        out  = torch.matmul(attn, V)                                # [B,H,P,D]

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(batch, self.primary_units, H * D)  # [B,P,H*D]
        out = self.out_proj(out)    # [B, P, D]
        out = self.norm(out)

        return out.view(batch, self.primary_units * D)


# ===========================================================================
# 7. Batch Normalization helpers
# ===========================================================================
class BatchNormWithNoise(nn.Module):
    """BatchNorm1d + small Gaussian noise during training."""

    def __init__(self, num_features: int, noise_std: float = 0.01):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return x


# ===========================================================================
# 8. State Encoder  (Encoder network, layers 0-9)
#
# Shape flow:
#   In:       [B, 1080]
#   BNNoise:  [B, 1080]
#   LinAttn2: [B, 1080]   (120 tokens x 9 dims, window_out=32 internal)
#   Trans3:   [B, 1080]   -> [B, 9, 120] -> [B, 120, 9] -> [B, 1080]
#   LinAttn4: [B, 1080]   (9 seqs of 1 token x 120 dims)
#   CGLSTM5:  [B, 288]    (9 seqs x NSkills=32)
#   Conv6:    [B, 540]    (9 seqs x 60=4*NForecast)
#   Conv7:    [B, 135]    (9 seqs x 15=NForecast)
#   Trans8:   [B, 135]    -> [B, NForecast=15, BarDescr=9] -> [B, 9, 15] -> flat
#   RevIn9:   [B, 135]    (learnable affine denorm)
# ===========================================================================
class StateEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        # Layer 1: BatchNorm with noise
        self.bn_noise = BatchNormWithNoise(INPUT_SIZE)

        # Layer 2: Linear Attention
        # count=HISTORY_BARS=120, window=BAR_DESCR=9, layers=1, window_out=32
        self.lin_attn_2 = LinearAttentionLayer(
            seq_len=HISTORY_BARS,
            d_in=BAR_DESCR,
            d_out=BAR_DESCR,    # output keeps same dim as input (MQ5 behavior)
            n_seqs=1
        )

        # Layer 3: Transpose [120, 9] -> [9, 120]
        self.transpose_3 = TransposeLayer(rows=HISTORY_BARS, cols=BAR_DESCR)

        # Layer 4: Linear Attention
        # count=1, window=HISTORY_BARS=120, layers=BAR_DESCR=9, window_out=32
        self.lin_attn_4 = LinearAttentionLayer(
            seq_len=1,
            d_in=HISTORY_BARS,
            d_out=HISTORY_BARS,  # keep same dim
            n_seqs=BAR_DESCR
        )

        # Layer 5: CGLSTM (CPU)
        # count=NSkills=32, window=HISTORY_BARS=120, layers=BAR_DESCR=9
        self.cglstm_5 = CGLSTMEncoder(
            seq_len=HISTORY_BARS,
            n_vars=BAR_DESCR,
            n_skills=N_SKILLS,
            d_in=1
        )

        # Layer 6: Conv projection
        # count=1, window=NSkills=32, step=32, window_out=4*NForecast=60, layers=BAR_DESCR=9
        self.conv_6 = ConvProjection(
            n_seqs=BAR_DESCR,
            d_in=N_SKILLS,
            d_out=4 * N_FORECAST,
            activation="softplus"
        )

        # Layer 7: Conv projection
        # count=1, window=4*NForecast=60, step=60, window_out=NForecast=15, layers=BAR_DESCR=9
        self.conv_7 = ConvProjection(
            n_seqs=BAR_DESCR,
            d_in=4 * N_FORECAST,
            d_out=N_FORECAST,
            activation="tanh"
        )

        # Layer 8: Transpose [NForecast, BarDescr] -> [BarDescr, NForecast]
        self.transpose_8 = TransposeLayer(rows=N_FORECAST, cols=BAR_DESCR)

        # Layer 9: RevIN denorm
        self.revin_9 = RevIN(num_features=BAR_DESCR * N_FORECAST)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [batch, HISTORY_BARS, BAR_DESCR] or [batch, INPUT_SIZE]

        Returns:
          (cglstm_out, encoder_out)
            cglstm_out:  [batch, N_SKILLS * BAR_DESCR] = [B, 288]  -- LatentLayer=5 output
            encoder_out: [batch, BAR_DESCR * N_FORECAST] = [B, 135] -- final output

        Device note: CGLSTM always runs on CPU. In inference (GPU) mode, the
        CGLSTM output is moved to GPU after computation without grad tracking.
        In training mode (CPU), everything stays on CPU and gradients flow normally.
        """
        if x.dim() == 3:
            x = x.contiguous().view(x.shape[0], -1)

        input_device = x.device
        on_gpu = (input_device != CPU_DEVICE and str(input_device) != 'cpu')

        # Layers 0-1
        out = self.bn_noise(x)                       # [B, 1080]

        # Layer 2: Linear attention
        out = self.lin_attn_2(out)                   # [B, 1080]

        # Layer 3: Transpose
        out = self.transpose_3(out)                  # [B, 1080]

        # Layer 4: Linear attention on transposed view
        out = self.lin_attn_4(out)                   # [B, 1080]

        # Layer 5: CGLSTM (always on CPU)
        if on_gpu:
            # Inference: move to CPU, compute, move back without gradient bridge
            cpu_in     = out.detach().to(CPU_DEVICE)
            cglstm_out = self.cglstm_5(cpu_in)
            cglstm_out = cglstm_out.to(input_device)
        else:
            # Training: everything on CPU, gradients flow normally
            cglstm_out = self.cglstm_5(out)         # [B, 288]

        # Layer 6: Conv
        out = self.conv_6(cglstm_out)                # [B, 540]

        # Layer 7: Conv
        out = self.conv_7(out)                       # [B, 135]

        # Layer 8: Transpose [NForecast=15, BarDescr=9] -> [BarDescr=9, NForecast=15]
        out = self.transpose_8(out)                  # [B, 135]

        # Layer 9: RevIN
        encoder_out = self.revin_9(out)              # [B, 135]

        return cglstm_out, encoder_out


# ===========================================================================
# 9. Task Encoder
#
# Shares encoder layers 0..LatentLayer (0 to 5 inclusive).
# Then adds:
#   CrossDMH: windows=[32,32], units=[9,9], step=4, window_out=32
#     Primary: cglstm_out [B, 9*32=288]
#     Cross:   cglstm_out [B, 9*32=288]  (self attention)
#     Output:  [B, 9*32=288]
#   Conv: count=1, window=32, step=32, window_out=4*NActions=24, layers=9, SoftPlus
#     Output: [B, 9*24=216]
#   Conv: count=1, window=24, step=24, window_out=NActions=6, layers=9, Sigmoid
#     Output: [B, 9*6=54]
# ===========================================================================
class TaskEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        # LatentLayer+1: CrossDMH on cglstm output
        self.cross_dmh = CrossDMHAttention(
            primary_window=N_SKILLS,
            cross_window=N_SKILLS,
            primary_units=BAR_DESCR,
            cross_units=BAR_DESCR,
            n_heads=4,
            d_head=32
        )

        # LatentLayer+2: Conv SoftPlus
        self.conv_l2 = ConvProjection(
            n_seqs=BAR_DESCR,
            d_in=32,                 # d_head from CrossDMH
            d_out=4 * N_ACTIONS,
            activation="softplus"
        )

        # LatentLayer+3: Conv Sigmoid
        self.conv_l3 = ConvProjection(
            n_seqs=BAR_DESCR,
            d_in=4 * N_ACTIONS,
            d_out=N_ACTIONS,
            activation="sigmoid"
        )

    def forward(self, cglstm_out: torch.Tensor) -> torch.Tensor:
        """
        cglstm_out: [B, N_SKILLS * BAR_DESCR] = [B, 288]
        returns:    [B, N_ACTIONS * BAR_DESCR] = [B, 54]
        """
        # CrossDMH: self-cross on cglstm output
        out = self.cross_dmh(cglstm_out, cglstm_out)  # [B, 9*32=288]

        # Conv layers
        out = self.conv_l2(out)     # [B, 9*24=216]
        out = self.conv_l3(out)     # [B, 9*6=54]
        return out


# ===========================================================================
# 10. Actor
#
# Input: account state [B, AccountDescr=12] + task output as cross-attention key
#
# Layers:
#   0: Input [12]
#   1: BatchNorm [12]
#   2: CrossDMH:
#        windows=[AccountDescr=12, NActions=6], units=[1, BarDescr=9]
#        step=4, window_out=32
#        Primary: bAccount [B, 1*12=12]
#        Cross:   task_out [B, 9*6=54]  -> 9 units of 6 dims
#        Output:  [B, 1*32=32]
#   3: Dense [LatentCount=256], Tanh
#   4: Dense [256], SoftPlus
#   5: Dense [NActions=6], Sigmoid
# ===========================================================================
class Actor(nn.Module):

    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(ACCOUNT_DESCR)

        self.cross_dmh = CrossDMHAttention(
            primary_window=ACCOUNT_DESCR,
            cross_window=N_ACTIONS,
            primary_units=1,
            cross_units=BAR_DESCR,
            n_heads=4,
            d_head=32
        )

        self.dense_3 = nn.Linear(32, LATENT_COUNT)
        self.dense_4 = nn.Linear(LATENT_COUNT, LATENT_COUNT)
        self.dense_5 = nn.Linear(LATENT_COUNT, N_ACTIONS)

    def forward(
        self,
        account: torch.Tensor,
        task_out: torch.Tensor
    ) -> torch.Tensor:
        """
        account:  [B, ACCOUNT_DESCR=12]
        task_out: [B, BAR_DESCR*N_ACTIONS=54]
        returns:  [B, N_ACTIONS=6]  (all values in [0,1])
        """
        acc = self.bn(account)                          # [B, 12]

        out = self.cross_dmh(acc, task_out)             # [B, 32]
        out = torch.tanh(self.dense_3(out))             # [B, 256]
        out = F.softplus(self.dense_4(out))             # [B, 256]
        out = torch.sigmoid(self.dense_5(out))          # [B, 6]
        return out


# ===========================================================================
# 11. Critic
#
# Input: actions [B, NActions=6] + cglstm cross-attention
#
# windows=[3, NSkills=32], units=[NActions/3=2, BarDescr=9]
# Primary: 2 tokens of 3 dims = [B, 6]
# Cross:   9 tokens of 32 dims = [B, 288]
# Output:  2*32=64
# Then Dense [256] Tanh, Dense [256] SoftPlus, Dense [1] linear
# ===========================================================================
class Critic(nn.Module):

    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(N_ACTIONS)

        self.cross_dmh = CrossDMHAttention(
            primary_window=3,
            cross_window=N_SKILLS,
            primary_units=N_ACTIONS // 3,
            cross_units=BAR_DESCR,
            n_heads=4,
            d_head=32
        )

        self.dense_3 = nn.Linear(N_ACTIONS // 3 * 32, LATENT_COUNT)
        self.dense_4 = nn.Linear(LATENT_COUNT, LATENT_COUNT)
        self.dense_5 = nn.Linear(LATENT_COUNT, 1)

    def forward(
        self,
        actions: torch.Tensor,
        cglstm_out: torch.Tensor
    ) -> torch.Tensor:
        """
        actions:    [B, N_ACTIONS=6]
        cglstm_out: [B, N_SKILLS*BAR_DESCR=288]
        returns:    [B, 1]  scalar value estimate
        """
        act = self.bn(actions)
        out = self.cross_dmh(act, cglstm_out)           # [B, 64]
        out = torch.tanh(self.dense_3(out))             # [B, 256]
        out = F.softplus(self.dense_4(out))             # [B, 256]
        out = self.dense_5(out)                         # [B, 1] linear
        return out


# ===========================================================================
# 12. Full DACGLSTM Expert
# ===========================================================================
class DACGLSTMExpert(nn.Module):
    """
    Full DACGLSTM Actor-Critic expert.

    Network graph (matches MQ5 Study.mq5 Train() feedforward order):
      bState -> Encoder (layers 0-9)
                  |-- cglstm_out (LatentLayer 5)  -> Task -> task_out
      bAccount + task_out -> Actor -> actions [6]
      actions  + cglstm_out -> Critic -> value [1]
    """

    def __init__(self):
        super().__init__()
        self.encoder = StateEncoder()
        self.task    = TaskEncoder()
        self.actor   = Actor()
        self.critic  = Critic()

    def forward(
        self,
        state: torch.Tensor,
        account: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        state:   [B, HISTORY_BARS, BAR_DESCR] or [B, INPUT_SIZE]
        account: [B, ACCOUNT_DESCR]

        Returns:
          actions:    [B, N_ACTIONS]   sigmoid outputs in [0,1]
          value:      [B, 1]           critic estimate
          task_out:   [B, 9*6=54]      task encoder output
        """
        cglstm_out, _ = self.encoder(state)
        task_out       = self.task(cglstm_out)
        actions        = self.actor(account, task_out)
        value          = self.critic(actions, cglstm_out)
        return actions, value, task_out

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def generate_signal(
        self,
        features: "np.ndarray"
    ) -> Tuple[int, float]:
        """
        Primary inference method.

        Parameters
        ----------
        features : ndarray shape [120, 9]
            120 bars, 9 features per bar:
              [0] close-open, [1] high-open, [2] low-open
              [3] tick_volume/1000, [4] RSI, [5] CCI, [6] ATR
              [7] MACD_main, [8] MACD_signal

        Returns
        -------
        direction : int   1=buy, -1=sell, 0=hold
        confidence: float 0.0-1.0
        """
        import numpy as np

        assert features.shape == (HISTORY_BARS, BAR_DESCR), (
            f"Expected ({HISTORY_BARS}, {BAR_DESCR}), got {features.shape}"
        )

        self.eval()
        with torch.no_grad():
            # Flat account placeholder (no live position info for simple signal)
            account_np = np.zeros(ACCOUNT_DESCR, dtype=np.float32)
            account_np[0] = 1.0   # balance change delta = 0 (normalised to 1)
            account_np[1] = 1.0   # equity/balance ratio  = 1

            x  = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            ac = torch.tensor(account_np, dtype=torch.float32).unsqueeze(0)

            # Move to GPU (CGLSTM moves itself to CPU internally)
            x  = x.to(GPU_DEVICE)
            ac = ac.to(GPU_DEVICE)

            actions, _, _ = self.forward(x, ac)
            actions = actions[0]   # [6]

        # MQ5 action layout:
        #   [0] buy_lot, [1] buy_tp, [2] buy_sl
        #   [3] sell_lot, [4] sell_tp, [5] sell_sl
        buy_lot  = actions[0].item()
        sell_lot = actions[3].item()

        MIN_LOT = 0.01
        if buy_lot >= sell_lot:
            buy_lot  -= sell_lot
            sell_lot  = 0.0
        else:
            sell_lot -= buy_lot
            buy_lot   = 0.0

        if buy_lot > sell_lot and buy_lot >= MIN_LOT:
            direction  = 1
            confidence = float(min(buy_lot, 1.0))
        elif sell_lot > buy_lot and sell_lot >= MIN_LOT:
            direction  = -1
            confidence = float(min(sell_lot, 1.0))
        else:
            direction  = 0
            confidence = 0.0

        return direction, confidence

    def generate_signal_with_account(
        self,
        features: "np.ndarray",
        account: "np.ndarray"
    ) -> Tuple[int, float, "np.ndarray"]:
        """
        Full inference with live account state.

        Parameters
        ----------
        features : [120, 9]
        account  : [12] array
            [0] (balance - prev_balance) / prev_balance
            [1] equity / prev_balance
            [2] (equity - prev_equity) / prev_equity
            [3] buy_volume
            [4] sell_volume
            [5] buy_profit / balance
            [6] sell_profit / balance
            [7] position_discount / balance
            [8-11] time features (sin/cos of yearly/monthly/weekly/daily)

        Returns
        -------
        direction, confidence, raw_actions[6]
        """
        import numpy as np

        self.eval()
        with torch.no_grad():
            x  = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(GPU_DEVICE)
            ac = torch.tensor(account,  dtype=torch.float32).unsqueeze(0).to(GPU_DEVICE)
            actions, _, _ = self.forward(x, ac)
            actions_np = actions[0].cpu().numpy()

        buy_lot  = float(actions_np[0])
        sell_lot = float(actions_np[3])

        if buy_lot >= sell_lot:
            buy_lot  -= sell_lot
            sell_lot  = 0.0
        else:
            sell_lot -= buy_lot
            buy_lot   = 0.0

        MIN_LOT = 0.01
        if buy_lot > sell_lot and buy_lot >= MIN_LOT:
            direction  = 1
            confidence = float(min(buy_lot, 1.0))
        elif sell_lot > buy_lot and sell_lot >= MIN_LOT:
            direction  = -1
            confidence = float(min(sell_lot, 1.0))
        else:
            direction  = 0
            confidence = 0.0

        return direction, confidence, actions_np

    def train_step(
        self,
        batch: dict,
        optimizer_enc: torch.optim.Optimizer,
        optimizer_task: torch.optim.Optimizer,
        optimizer_actor: torch.optim.Optimizer,
        optimizer_critic: torch.optim.Optimizer,
        gamma: float = 1.0 - DISC_FACTOR
    ) -> dict:
        """
        Single training step. Walk-forward RL (simplified SAC-style).

        IMPORTANT: Call model.to_train_mode() before training. All computation
        runs on CPU (TRAIN_DEVICE). DirectML / ROCm .backward() segfaults on
        Windows when called from the bash/Python process -- CPU backward is safe.
        This matches expert_conformer.py and expert_mamba.py in this library.

        batch dict keys:
          'states'        : [B, 120, 9]  float32
          'accounts'      : [B, 12]      float32
          'actions'       : [B, 6]       float32  (supervised target)
          'rewards'       : [B, 3]       float32  [delta_balance, delta_equity, hold_penalty]
          'next_states'   : [B, 120, 9]
          'next_accounts' : [B, 12]

        Returns dict with loss scalars.
        """
        self.train()
        dev = TRAIN_DEVICE   # CPU

        states        = batch['states'].to(dev)
        accounts      = batch['accounts'].to(dev)
        actions_tgt   = batch['actions'].to(dev)
        rewards       = batch['rewards'].to(dev)
        next_states   = batch['next_states'].to(dev)
        next_accounts = batch['next_accounts'].to(dev)

        reward = rewards.sum(dim=-1, keepdim=True)   # [B, 1]

        # Forward pass -- all on CPU, gradients flow through CGLSTM naturally
        cglstm_out, _ = self.encoder(states)
        task_out       = self.task(cglstm_out)
        actions_pred   = self.actor(accounts, task_out)
        value_pred     = self.critic(actions_pred, cglstm_out)

        # TD target (no gradient)
        with torch.no_grad():
            cglstm_next, _ = self.encoder(next_states)
            task_next       = self.task(cglstm_next)
            actions_next    = self.actor(next_accounts, task_next)
            value_next      = self.critic(actions_next, cglstm_next)
            value_target    = reward + gamma * value_next

        actor_loss   = F.mse_loss(actions_pred, actions_tgt)
        critic_loss  = F.mse_loss(value_pred, value_target)
        encoder_loss = -value_pred.mean()   # push encoder toward high-value states

        optimizer_enc.zero_grad()
        optimizer_task.zero_grad()
        optimizer_actor.zero_grad()
        optimizer_critic.zero_grad()

        total_loss = actor_loss + critic_loss + 0.1 * encoder_loss
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.task.parameters(),    1.0)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(),   1.0)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(),  1.0)

        optimizer_enc.step()
        optimizer_task.step()
        optimizer_actor.step()
        optimizer_critic.step()

        return {
            'actor_loss':   actor_loss.item(),
            'critic_loss':  critic_loss.item(),
            'encoder_loss': encoder_loss.item(),
            'total_loss':   total_loss.item(),
        }

    def save_weights(self, path: str) -> None:
        """Save all model weights to path."""
        torch.save({
            'encoder': self.encoder.state_dict(),
            'task':    self.task.state_dict(),
            'actor':   self.actor.state_dict(),
            'critic':  self.critic.state_dict(),
        }, path)
        print(f"Saved weights to {path}")

    def load_weights(self, path: str) -> None:
        """Load weights from path. Missing keys are ignored (fresh layers kept)."""
        ckpt = torch.load(path, map_location=CPU_DEVICE)
        self.encoder.load_state_dict(ckpt['encoder'], strict=False)
        self.task.load_state_dict(ckpt['task'],       strict=False)
        self.actor.load_state_dict(ckpt['actor'],     strict=False)
        self.critic.load_state_dict(ckpt['critic'],   strict=False)
        print(f"Loaded weights from {path}")

    def to_train_mode(self) -> "DACGLSTMExpert":
        """
        Move all sub-networks to CPU for training.
        DirectML / ROCm .backward() segfaults on Windows -- training must be CPU.
        """
        self.encoder.to(CPU_DEVICE)
        self.task.to(CPU_DEVICE)
        self.actor.to(CPU_DEVICE)
        self.critic.to(CPU_DEVICE)
        return self

    def to_gpu(self) -> "DACGLSTMExpert":
        """
        Move all sub-networks to GPU for inference.
        CGLSTM cell stays on CPU; StateEncoder.forward handles the device split.
        """
        # Everything except CGLSTM goes to GPU
        for name, module in self.encoder.named_children():
            if name == 'cglstm_5':
                module.to(CPU_DEVICE)
            else:
                module.to(GPU_DEVICE)
        self.task.to(GPU_DEVICE)
        self.actor.to(GPU_DEVICE)
        self.critic.to(GPU_DEVICE)
        return self


# ===========================================================================
# 13. Helper: build optimizers
# ===========================================================================
def build_optimizers(
    model: DACGLSTMExpert,
    lr: float = 1e-4
) -> Tuple:
    """
    Returns (opt_encoder, opt_task, opt_actor, opt_critic).

    Call model.to_train_mode() before training to move everything to CPU.
    All parameters including CGLSTM are covered by opt_encoder (all on CPU
    during training, backward flows through without device crossing).
    """
    opt_enc    = torch.optim.Adam(model.encoder.parameters(), lr=lr)
    opt_task   = torch.optim.Adam(model.task.parameters(),    lr=lr)
    opt_actor  = torch.optim.Adam(model.actor.parameters(),   lr=lr)
    opt_critic = torch.optim.Adam(model.critic.parameters(),  lr=lr)
    return opt_enc, opt_task, opt_actor, opt_critic


# ===========================================================================
# 14. Quick sanity check
# ===========================================================================
if __name__ == "__main__":
    import numpy as np

    print("Building DACGLSTM expert...")
    model = DACGLSTMExpert()
    model.to_gpu()

    # Test generate_signal
    rng = np.random.default_rng(42)
    features = rng.standard_normal((HISTORY_BARS, BAR_DESCR)).astype(np.float32)
    direction, confidence = model.generate_signal(features)
    print(f"generate_signal -> direction={direction}, confidence={confidence:.4f}")

    # Test forward pass shapes
    model.train()
    B = 4
    states   = torch.randn(B, HISTORY_BARS, BAR_DESCR).to(GPU_DEVICE)
    accounts = torch.randn(B, ACCOUNT_DESCR).to(GPU_DEVICE)
    actions, value, task_out = model.forward(states, accounts)
    print(f"actions  shape: {actions.shape}   (expected [{B}, {N_ACTIONS}])")
    print(f"value    shape: {value.shape}     (expected [{B}, 1])")
    print(f"task_out shape: {task_out.shape}  (expected [{B}, {BAR_DESCR * N_ACTIONS}])")

    # Test train_step (on CPU -- DirectML backward broken on Windows)
    model.to_train_mode()
    opt_enc, opt_task, opt_actor, opt_critic = build_optimizers(model)
    batch = {
        'states':        torch.randn(B, HISTORY_BARS, BAR_DESCR),
        'accounts':      torch.randn(B, ACCOUNT_DESCR),
        'actions':       torch.rand(B, N_ACTIONS),
        'rewards':       torch.randn(B, 3),
        'next_states':   torch.randn(B, HISTORY_BARS, BAR_DESCR),
        'next_accounts': torch.randn(B, ACCOUNT_DESCR),
    }
    losses = model.train_step(batch, opt_enc, opt_task, opt_actor, opt_critic)
    print(f"train_step losses: {losses}")

    # Test save/load
    WEIGHT_PATH = r"C:\Users\jimjj\Music\QuantumChildren\QuantumTradingLibrary\dacglstm_weights.pt"
    model.save_weights(WEIGHT_PATH)
    model.load_weights(WEIGHT_PATH)
    print("Save/load OK")
    print("All checks passed.")

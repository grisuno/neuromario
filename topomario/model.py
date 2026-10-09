#!/usr/bin/env python3
"""
TopoMario: Complex-valued spectral model for playing Super Mario Bros.

Based on TopoGPT2 architecture (quaternion spectral layers) but adapted
for game state processing and action generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt
from safetensors.torch import save_file as st_save, load_file as st_load
import numpy as np
import math
import os
import sys
import time
from pathlib import Path
import json
import hashlib
import logging
import warnings
import argparse
from datetime import datetime
from typing import Dict, Tuple, Optional, List, Any, Union
from dataclasses import dataclass, asdict
from collections import deque

warnings.filterwarnings('ignore')


@dataclass
class TopoMarioConfig:
    """Configuration for TopoMario model."""

    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    RANDOM_SEED: int = 42
    USE_AMP: bool = True

    SCALE: str = 'small'

    # Game state dimensions
    SCREEN_HEIGHT: int = 240
    SCREEN_WIDTH: int = 256
    # 4 frames apilados x 3 canales = 12: la red ve velocidad, no fotos.
    # (checkpoints viejos de 3ch reutilizan todo menos el stem visual.)
    SCREEN_CHANNELS: int = 12
    FRAME_STACK: int = 4
    SEMANTIC_DIM: int = 33  # state + coin/powerup/enemy/block/secret/damage events

    # Action space (SIMPLE_MOVEMENT from gym-super-mario-bros)
    N_ACTIONS: int = 7

    # Model architecture
    D_MODEL: int = 256
    N_HEADS: int = 8
    N_KV_HEADS: int = 0
    N_LAYERS: int = 6
    DROPOUT: float = 0.1

    # MoE
    MOE_ENABLED: bool = True
    N_EXPERTS: int = 4
    MOE_TOP_K: int = 2
    MOE_AUX_LOSS_WEIGHT: float = 0.01

    # Torus topology
    TORUS_GRID_SIZE: int = 8
    TORUS_RADIAL_BINS: int = 2
    TORUS_ANGULAR_BINS: int = 4

    # Spectral / Autoencoder
    SPECTRAL_LATENT_RATIO: float = 0.5
    SPECTRAL_KERNEL_INIT_SCALE: float = 0.02
    NUM_SPECTRAL_LAYERS: int = 2
    AE_RECON_WEIGHT: float = 0.01

    # Neuronas liquidas + memoria episodica (neurologos)
    LIQUID_PLASTICITY: float = 0.1
    EPISODIC_PLASTICITY: float = 1.0

    # Training
    BATCH_SIZE: int = 4
    GRAD_ACCUM_STEPS: int = 8
    LEARNING_RATE: float = 3e-4
    WEIGHT_DECAY: float = 0.1
    EPOCHS: int = 10
    WARMUP_RATIO: float = 0.05
    GRADIENT_CLIP_NORM: float = 1.0
    GRADIENT_CHECKPOINTING: bool = True

    # Sliding Window Attention
    ATTN_WINDOW: int = 0

    # Latent Memory Tokens
    N_MEMORY_TOKENS: int = 0
    MEMORY_SEGMENT_LEN: int = 0

    # Progressive Sequence Length
    PROGRESSIVE_SEQ: bool = True
    PROGRESSIVE_SEQ_STEPS: Tuple[int, int, int] = (128, 256, 512)
    PROGRESSIVE_SEQ_EPOCHS: Tuple[int, int, int] = (3, 3, 4)

    # Quantization
    QUANTIZE_EMBEDDINGS: bool = True
    QUANTIZE_FFN: bool = False
    QUANT_EMBED_BITS: int = 8

    # Checkpoints
    CHECKPOINT_DIR: str = 'checkpoints_topomario'
    CHECKPOINT_INTERVAL_MINUTES: int = 10
    MAX_CHECKPOINTS: int = 5

    # Logging
    LOG_INTERVAL_STEPS: int = 100
    EVAL_INTERVAL_STEPS: int = 500
    LOG_LEVEL: str = 'INFO'

    def __post_init__(self):
        presets = {
            'micro':  dict(D_MODEL=64,  N_HEADS=4,  N_LAYERS=2,  ATTN_WINDOW=64, N_MEMORY_TOKENS=16, MEMORY_SEGMENT_LEN=64, NUM_SPECTRAL_LAYERS=1),
            'small':  dict(D_MODEL=256, N_HEADS=8,  N_LAYERS=6,  ATTN_WINDOW=128, N_MEMORY_TOKENS=32, MEMORY_SEGMENT_LEN=128, NUM_SPECTRAL_LAYERS=2),
            'medium': dict(D_MODEL=512, N_HEADS=8,  N_LAYERS=12, ATTN_WINDOW=256, N_MEMORY_TOKENS=64, MEMORY_SEGMENT_LEN=256, NUM_SPECTRAL_LAYERS=2),
            'large':  dict(D_MODEL=512, N_HEADS=8,  N_LAYERS=13, ATTN_WINDOW=256, N_MEMORY_TOKENS=64, MEMORY_SEGMENT_LEN=256, NUM_SPECTRAL_LAYERS=3),
            'gpt2':   dict(D_MODEL=768, N_HEADS=12, N_LAYERS=12, ATTN_WINDOW=256, N_MEMORY_TOKENS=64, MEMORY_SEGMENT_LEN=256, NUM_SPECTRAL_LAYERS=3),
            # single.py presets: mas grandes porque small no aprendia ni en una noche.
            # xl/xxl suben capacidad y profundidad espectral (mas Fourier 2D en toro).
            'xl':     dict(D_MODEL=768, N_HEADS=12, N_LAYERS=16, ATTN_WINDOW=256, N_MEMORY_TOKENS=64, MEMORY_SEGMENT_LEN=256, NUM_SPECTRAL_LAYERS=4, MOE_ENABLED=True, N_EXPERTS=8, MOE_TOP_K=2),
            'xxl':    dict(D_MODEL=1024, N_HEADS=16, N_LAYERS=16, ATTN_WINDOW=256, N_MEMORY_TOKENS=64, MEMORY_SEGMENT_LEN=256, NUM_SPECTRAL_LAYERS=4, MOE_ENABLED=True, N_EXPERTS=8, MOE_TOP_K=2),
        }
        if self.SCALE in presets:
            for k, v in presets[self.SCALE].items():
                setattr(self, k, v)

        assert self.D_MODEL % 4 == 0, "D_MODEL must be divisible by 4 (quaternions)"
        assert self.D_MODEL % self.N_HEADS == 0, "D_MODEL must be divisible by N_HEADS"
        self.D_QUAT = self.D_MODEL // 4
        self.D_HEAD = self.D_MODEL // self.N_HEADS
        self.SPECTRAL_LATENT_DIM = max(16, int(self.D_MODEL * self.SPECTRAL_LATENT_RATIO))
        self.N_TORUS_NODES = self.TORUS_RADIAL_BINS * self.TORUS_ANGULAR_BINS
        if self.N_KV_HEADS == 0:
            kv = max(1, self.N_HEADS // 4)
            while self.N_HEADS % kv != 0:
                kv -= 1
            self.N_KV_HEADS = kv
        elif self.N_KV_HEADS == -1:
            self.N_KV_HEADS = self.N_HEADS
        assert self.N_HEADS % self.N_KV_HEADS == 0, \
            "N_HEADS must be divisible by N_KV_HEADS"
        self.GQA_GROUPS = self.N_HEADS // self.N_KV_HEADS


def setup_logger(name: str, level: str = 'INFO') -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s'))
        logger.addHandler(h)
    return logger


def set_seed(seed: int, device: str):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if 'cuda' in device and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


# ============================================================================
# LIQUID NEURONS + EPISODIC MEMORY (port de neurologos: LiquidNeuron de
# neurologos.py + MicroContinuumCell de neurologos_cpu_v7.py)
#
# - LiquidNeuron: via lenta (Linear) + via rapida plastica (matriz que se
#   mueve con ruido hebbiano en train). Constante de tiempo adaptativa.
# - EpisodicMemoryCell: matriz semantic_memory con update Hebbiano
#   (outer-product, sin gradiente: funciona con PPO aunque se mezcle el
#   batch, no necesita BPTT). La compuerta mezcla via lenta vs recuerdo.
# ============================================================================

class LiquidNeuron(nn.Module):
    """Neurona liquida: slow path + fast plastic path (neurologos)."""

    def __init__(self, dim: int):
        super().__init__()
        self.slow = nn.Linear(dim, dim)
        self.fast = nn.Parameter(torch.zeros(dim, dim))
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, plasticity: float = 0.1) -> torch.Tensor:
        if self.training and plasticity > 0:
            with torch.no_grad():
                delta = torch.randn_like(self.fast) * plasticity
                self.fast.data = 0.95 * self.fast.data + 0.05 * delta
        out = self.slow(x) + F.linear(x, self.fast)
        return self.ln(torch.tanh(out))


class EpisodicMemoryCell(nn.Module):
    """Memoria episodica Hebbiana (MicroContinuumCell de neurologos).

    Guarda correlaciones estado->estado en semantic_memory con outer
    product. El update es no_grad (Hebb local), asi que sobrevive al
    shuffle de PPO: no es BPTT, es plasticidad local como en neurologos.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.V_slow = nn.Linear(dim, dim, bias=False)
        nn.init.orthogonal_(self.V_slow.weight, gain=0.1)
        self.gate_net = nn.Linear(dim, 1)
        self.semantic_memory = nn.Parameter(torch.zeros(dim, dim))
        nn.init.normal_(self.semantic_memory, std=0.01)

    def forward(self, x: torch.Tensor, plasticity: float = 1.0) -> torch.Tensor:
        v = self.V_slow(x)
        v = torch.clamp(v, -2.0, 2.0)
        y_pred = F.linear(x, self.semantic_memory)
        y_pred = torch.clamp(y_pred, -2.0, 2.0)
        v_pred = self.V_slow(y_pred)
        gate = torch.sigmoid(self.gate_net(v.detach())) * plasticity
        if self.training and plasticity > 0:
            with torch.no_grad():
                # Hebb: correlacion del batch como traza episodica
                delta = torch.bmm(x.unsqueeze(-1), x.unsqueeze(1))
                self.semantic_memory.data = (
                    0.95 * self.semantic_memory.data
                    + 0.01 * delta.mean(dim=0))
                mem_norm = self.semantic_memory.data.norm().clamp(min=1e-6)
                self.semantic_memory.data = (
                    self.semantic_memory.data / mem_norm * 0.5)
        output = gate * v + (1 - gate) * v_pred
        return torch.clamp(output, -2.0, 2.0)


# ============================================================================
# QUATERNION ALGEBRA
# ============================================================================

class QuaternionOps:
    """Pure quaternion operations in PyTorch."""

    @staticmethod
    def hamilton_product(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dim=-1)

    @staticmethod
    def normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return q / (q.norm(dim=-1, keepdim=True) + eps)

    @staticmethod
    def conjugate(q: torch.Tensor) -> torch.Tensor:
        sign = q.new_tensor([1, -1, -1, -1])
        return q * sign

    @staticmethod
    def rotate_vector(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros(*v.shape[:-1], 1, device=v.device, dtype=v.dtype)
        v_q = torch.cat([zero, v], dim=-1)
        q_c = QuaternionOps.conjugate(q)
        rotated = QuaternionOps.hamilton_product(
            QuaternionOps.hamilton_product(q, v_q), q_c)
        return rotated[..., 1:]


class QuaternionLinear(nn.Module):
    """Linear layer with quaternion weights."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        assert in_features % 4 == 0 and out_features % 4 == 0
        self.in_q = in_features // 4
        self.out_q = out_features // 4

        self.Ww = nn.Linear(self.in_q, self.out_q, bias=False)
        self.Wx = nn.Linear(self.in_q, self.out_q, bias=False)
        self.Wy = nn.Linear(self.in_q, self.out_q, bias=False)
        self.Wz = nn.Linear(self.in_q, self.out_q, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        for w in [self.Ww, self.Wx, self.Wy, self.Wz]:
            nn.init.normal_(w.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.in_q
        xw, xx, xy, xz = x[..., :d], x[..., d:2*d], x[..., 2*d:3*d], x[..., 3*d:]
        ow = self.Ww(xw) - self.Wx(xx) - self.Wy(xy) - self.Wz(xz)
        ox = self.Ww(xx) + self.Wx(xw) + self.Wy(xz) - self.Wz(xy)
        oy = self.Ww(xy) - self.Wx(xz) + self.Wy(xw) + self.Wz(xx)
        oz = self.Ww(xz) + self.Wx(xy) - self.Wy(xx) + self.Wz(xw)
        out = torch.cat([ow, ox, oy, oz], dim=-1)
        return out + self.bias if self.bias is not None else out


# ============================================================================
# SPECTRAL LAYER WITH QUATERNIONS
# ============================================================================

class QuaternionSpectralLayer(nn.Module):
    """2D spectral convolution with quaternions."""

    def __init__(self, in_q: int, out_q: int, grid_h: int, grid_w: int,
                 init_scale: float = 0.02):
        super().__init__()
        self.in_q = in_q
        self.out_q = out_q
        self.grid_h = grid_h
        self.grid_w = grid_w

        freq_h = grid_h
        freq_w = grid_w // 2 + 1

        for c in ('w', 'x', 'y', 'z'):
            self.register_parameter(f'kr_{c}',
                nn.Parameter(torch.randn(in_q, out_q, freq_h, freq_w) * init_scale))
            self.register_parameter(f'ki_{c}',
                nn.Parameter(torch.randn(in_q, out_q, freq_h, freq_w) * init_scale))

    def _kernel(self, c: str) -> torch.Tensor:
        return torch.complex(getattr(self, f'kr_{c}'), getattr(self, f'ki_{c}'))

    def _contract(self, W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        return torch.einsum('iohw,bihw->bohw', W, X)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.in_q
        xw, xx, xy, xz = x[:, :q], x[:, q:2*q], x[:, 2*q:3*q], x[:, 3*q:]

        Xw = torch.fft.rfft2(xw, s=(self.grid_h, self.grid_w))
        Xx = torch.fft.rfft2(xx, s=(self.grid_h, self.grid_w))
        Xy = torch.fft.rfft2(xy, s=(self.grid_h, self.grid_w))
        Xz = torch.fft.rfft2(xz, s=(self.grid_h, self.grid_w))

        Ww, Wx, Wy, Wz = self._kernel('w'), self._kernel('x'), self._kernel('y'), self._kernel('z')

        C = {}
        for wc, W in (('w', Ww), ('x', Wx), ('y', Wy), ('z', Wz)):
            for xc, X in (('w', Xw), ('x', Xx), ('y', Xy), ('z', Xz)):
                C[(wc, xc)] = self._contract(W, X)

        Pw = C[('w','w')] - C[('x','x')] - C[('y','y')] - C[('z','z')]
        Px = C[('w','x')] + C[('x','w')] + C[('y','z')] - C[('z','y')]
        Py = C[('w','y')] - C[('x','z')] + C[('y','w')] + C[('z','x')]
        Pz = C[('w','z')] + C[('x','y')] - C[('y','x')] + C[('z','w')]

        ow = torch.fft.irfft2(Pw, s=(self.grid_h, self.grid_w))
        ox = torch.fft.irfft2(Px, s=(self.grid_h, self.grid_w))
        oy = torch.fft.irfft2(Py, s=(self.grid_h, self.grid_w))
        oz = torch.fft.irfft2(Pz, s=(self.grid_h, self.grid_w))

        return torch.cat([ow, ox, oy, oz], dim=1)


# ============================================================================
# SPECTRAL AUTOENCODER
# ============================================================================

class SpectralAutoencoder(nn.Module):
    """Spectral autoencoder with quaternions."""

    def __init__(self, config: TopoMarioConfig):
        super().__init__()
        d = config.D_MODEL
        d_lat = config.SPECTRAL_LATENT_DIM
        d_q = config.D_QUAT
        g = config.TORUS_GRID_SIZE
        r = config.TORUS_RADIAL_BINS
        a = config.TORUS_ANGULAR_BINS
        init_s = config.SPECTRAL_KERNEL_INIT_SCALE

        n_freq = d // 2 + 1

        self.enc_kr = nn.Parameter(torch.randn(n_freq) * init_s)
        self.enc_ki = nn.Parameter(torch.randn(n_freq) * init_s)
        self.dec_kr = nn.Parameter(torch.randn(n_freq) * init_s)
        self.dec_ki = nn.Parameter(torch.randn(n_freq) * init_s)

        self.enc_proj = QuaternionLinear(d, d_lat)
        self.dec_proj = QuaternionLinear(d_lat, d)

        self.torus_spectral = nn.ModuleList([
            QuaternionSpectralLayer(d_q, d_q, r, a, init_scale=init_s)
            for _ in range(config.NUM_SPECTRAL_LAYERS)
        ])

        self.act = nn.GELU()
        self.d_model = d

    def _filter1d(self, x: torch.Tensor, kr: torch.Tensor, ki: torch.Tensor) -> torch.Tensor:
        X = torch.fft.rfft(x, dim=-1)
        K = torch.complex(kr, ki)
        return torch.fft.irfft(X * K, n=self.d_model, dim=-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_filt = self.act(self._filter1d(x, self.enc_kr, self.enc_ki))
        return self.enc_proj(x_filt)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.dec_proj(z)
        return self._filter1d(x, self.dec_kr, self.dec_ki)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        recon_loss = F.mse_loss(recon, x.detach())
        return z, recon_loss

    def process_torus_grid(self, grid: torch.Tensor) -> torch.Tensor:
        h = grid
        for layer in self.torus_spectral:
            h = self.act(layer(h))
        return h


# ============================================================================
# QUATERNION TORUS BRAIN
# ============================================================================

class QuaternionTorusBrain(nn.Module):
    """Replaces MLP in each transformer layer."""

    def __init__(self, d_model: int, config: TopoMarioConfig):
        super().__init__()
        self.d_model = d_model
        self.d_lat = config.SPECTRAL_LATENT_DIM
        self.d_q = d_model // 4
        self.n_radial = config.TORUS_RADIAL_BINS
        self.n_angular = config.TORUS_ANGULAR_BINS
        self.n_nodes = config.N_TORUS_NODES
        self.config = config

        self.spectral_ae = SpectralAutoencoder(config)

        self.latent_to_model = nn.Linear(self.d_lat, d_model)
        self.to_torus = nn.Linear(d_model, self.n_nodes * d_model)
        self.from_torus = nn.Linear(self.n_nodes * d_model, d_model)

        self.node_embed = nn.Parameter(torch.randn(self.n_nodes, d_model) * 0.02)

        self.message_pass = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(2)
        ])

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x_flat = x.reshape(B * S, D)

        z, recon_loss = self.spectral_ae(x_flat)
        z = self.latent_to_model(z)

        torus_in = self.to_torus(z)
        torus_in = torus_in.reshape(B * S, self.n_nodes, self.d_model)

        h = torus_in + self.node_embed.unsqueeze(0)

        for layer in self.message_pass:
            h = self.act(layer(h))
            h = h + torch.roll(h, shifts=1, dims=1)

        h = h.reshape(B * S, self.n_nodes * self.d_model)
        out = self.from_torus(h)

        out = out.reshape(B, S, D)
        return out, recon_loss


# ============================================================================
# ATTENTION MECHANISM
# ============================================================================

class SlidingWindowAttention(nn.Module):
    """Multi-head attention with sliding window."""

    def __init__(self, config: TopoMarioConfig):
        super().__init__()
        self.d_model = config.D_MODEL
        self.n_heads = config.N_HEADS
        self.n_kv_heads = config.N_KV_HEADS
        self.d_head = config.D_HEAD
        self.dropout = config.DROPOUT
        self.window = config.ATTN_WINDOW

        self.q_proj = nn.Linear(config.D_MODEL, config.D_MODEL)
        self.k_proj = nn.Linear(config.D_MODEL, config.D_MODEL)
        self.v_proj = nn.Linear(config.D_MODEL, config.D_MODEL)
        self.o_proj = nn.Linear(config.D_MODEL, config.D_MODEL)

        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, D = x.shape

        if S == 1:
            q = self.q_proj(x).reshape(B, 1, self.n_heads, self.d_head).transpose(1, 2)
            v = self.v_proj(x).reshape(B, 1, self.n_heads, self.d_head).transpose(1, 2)
            out = v
            out = out.transpose(1, 2).reshape(B, S, D)
            out = self.o_proj(out)
            return self.dropout_layer(out)

        q = self.q_proj(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).reshape(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).reshape(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)

        if self.n_kv_heads != self.n_heads:
            k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
            v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

        if self.window > 0 and S > self.window:
            mask = torch.ones(S, S, device=x.device, dtype=torch.bool)
            mask = torch.triu(mask, diagonal=self.window)
            mask = torch.tril(mask, diagonal=-self.window)
            mask = ~mask

        if mask is not None:
            attn_mask = torch.zeros(S, S, device=x.device)
            attn_mask = attn_mask.masked_fill(mask, float('-inf'))
        else:
            attn_mask = None

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0
        )

        out = out.transpose(1, 2).reshape(B, S, D)
        out = self.o_proj(out)
        return self.dropout_layer(out)


# ============================================================================
# TRANSFORMER LAYER
# ============================================================================

class TransformerLayer(nn.Module):
    """Single transformer layer with quaternion torus brain."""

    def __init__(self, config: TopoMarioConfig):
        super().__init__()
        self.config = config

        self.norm1 = nn.LayerNorm(config.D_MODEL)
        self.attn = SlidingWindowAttention(config)

        self.norm2 = nn.LayerNorm(config.D_MODEL)
        self.torus_brain = QuaternionTorusBrain(config.D_MODEL, config)

        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.norm1(x))

        torus_out, recon_loss = self.torus_brain(self.norm2(x))
        x = x + self.dropout(torus_out)

        return x, recon_loss


# ============================================================================
# GAME STATE ENCODER
# ============================================================================

class GameStateEncoder(nn.Module):
    """Encodes game state (screen + semantic data) into model dimension.

    MarioVision upgrade (ideas de neuromario/trimario4.py):

    - rama visual con mapa de saliency (attention 1x1 sobre conv, como
      ``VisualFeatureExtractor.attention_conv``) para que la red *vea a Mario*:
      el dashboard puede pintar el foco real en vez de una columna falsa.
    - decoder visual auxiliar (autoencoder sobre pantalla en gris 30x28) cuya
      recon loss se suma a la espectral: obliga a la vision a conservar
      geometria en vez de colapsar al vector semantico.
    - fusion por compuerta (estilo CorpusCallosum de neuromario): en vez de un
      solo Linear, una gate softmax de 2 vias mezcla visual/semantico segun
      contexto. Los pesos viejos (``fusion``) se reutilizan como rama base.

    Ojos multimodales (esta version):
    - ``vision_conv``: stem convolucional paralelo que NO colapsa a vector,
      deja tokens espaciales [B,240,D] (rejilla 15x16 sobre la pantalla).
    - ``vision_tf``: 2 capas transformer sobre esos tokens = razonamiento
      visual (relaciones espaciales: Mario vs tubo vs enemigo vs hueco).
    - ``cross_attn_*``: atencion cruzada vision<->semantico en ambas
      direcciones = fusion multimodal real, no solo concat+Linear.
    - ``cause_embed``: el PORQUE del paso anterior (vector causa de 16)
      entra como tercer modo, para que la red condicione en "me acaban de
      premiar por moneda / castigar por quieto".
    """

    FOCUS_H, FOCUS_W = 30, 32  # 240/8 x 256/8
    N_TOKENS_H, N_TOKENS_W = 15, 16
    N_CAUSES = 16

    def __init__(self, config: TopoMarioConfig):
        super().__init__()
        self.config = config

        self.screen_encoder = nn.Sequential(
            nn.Conv2d(config.SCREEN_CHANNELS, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, config.D_MODEL),
            nn.ReLU(),
        )

        self.semantic_encoder = nn.Sequential(
            nn.Linear(config.SEMANTIC_DIM, config.D_MODEL // 2),
            nn.ReLU(),
            nn.Linear(config.D_MODEL // 2, config.D_MODEL),
        )

        self.fusion = nn.Linear(config.D_MODEL * 2, config.D_MODEL)

        # --- MarioVision: saliency sobre la pantalla (downsample x8) ---
        self.saliency_net = nn.Sequential(
            nn.AvgPool2d(kernel_size=8, stride=8),  # [B,3,30,32]
            nn.Conv2d(config.SCREEN_CHANNELS, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        # compuerta visual<->semantico (corpus callosum de 2 vias)
        self.fusion_gate = nn.Sequential(
            nn.Linear(config.D_MODEL * 2, 64),
            nn.LayerNorm(64),
            nn.Tanh(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1),
        )
        self.visual_bias = nn.Linear(config.D_MODEL, config.D_MODEL)
        self.semantic_bias = nn.Linear(config.D_MODEL, config.D_MODEL)
        # decoder auxiliar: reconstruye gris 30x28 desde el feat visual
        self.vis_decoder = nn.Sequential(
            nn.Linear(config.D_MODEL, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, self.FOCUS_H * 28),
        )
        # --- ojos multimodales: tokens visuales + razonador + cross-modal ---
        d = config.D_MODEL
        nhead = config.N_HEADS if d % config.N_HEADS == 0 else 4
        self.vision_conv = nn.Sequential(
            nn.Conv2d(config.SCREEN_CHANNELS, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((self.N_TOKENS_H, self.N_TOKENS_W)),
            nn.Conv2d(64, d, kernel_size=1),
        )
        self.vision_pos = nn.Parameter(
            torch.randn(1, self.N_TOKENS_H * self.N_TOKENS_W, d) * 0.02)
        tf_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=d * 2,
            dropout=config.DROPOUT, batch_first=True)
        self.vision_tf = nn.TransformerEncoder(tf_layer, num_layers=2)
        # atencion cruzada en ambas direcciones (razonamiento multimodal)
        self.cross_vis2sem = nn.MultiheadAttention(
            d, nhead, dropout=config.DROPOUT, batch_first=True)
        self.cross_sem2vis = nn.MultiheadAttention(
            d, nhead, dropout=config.DROPOUT, batch_first=True)
        self.reason_norm = nn.LayerNorm(d)
        self.reason_mlp = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(), nn.Dropout(config.DROPOUT),
            nn.Linear(d * 2, d))
        # tercer modo: el PORQUE del paso anterior (16 causas binarias)
        self.cause_embed = nn.Linear(self.N_CAUSES, d)
        self._last_vis_attn = None

    def focus_map(self, screen: torch.Tensor) -> torch.Tensor:
        """Mapa de atencion [B,1,FOCUS_H,FOCUS_W] en [0,1]. No falla nunca.

        Combina saliency conv (donde hay sprite) con atencion del razonador
        visual (que tokens miro el cross-modal). Es el foco REAL de la red.
        """
        try:
            with torch.no_grad():
                sal = self.saliency_net(screen).detach()  # [B,1,FH,FW]
                attn = getattr(self, '_last_vis_attn', None)
                if attn is not None and attn.shape[0] == sal.shape[0]:
                    t = attn[:, 0, :]  # [B,T]
                    ht, wt = self.N_TOKENS_H, self.N_TOKENS_W
                    if t.shape[1] == ht * wt:
                        amap = t.view(-1, 1, ht, wt)
                        amap = amap / (amap.amax(dim=(2, 3), keepdim=True) + 1e-8)
                        grid = F.interpolate(
                            amap, size=(self.FOCUS_H, self.FOCUS_W),
                            mode='bilinear', align_corners=False)
                        return torch.clamp(0.5 * sal + 0.5 * grid, 0, 1)
                return sal
        except Exception:
            b = screen.shape[0]
            return torch.full((b, 1, self.FOCUS_H, self.FOCUS_W),
                              0.5, device=screen.device)

    def vision_reason(self, screen: torch.Tensor) -> torch.Tensor:
        """Tokens visuales razonados [B,T,D]. No falla: fallback a ceros."""
        try:
            fmap = self.vision_conv(screen)  # [B,D,Ht,Wt]
            b, dd, ht, wt = fmap.shape
            toks = fmap.flatten(2).transpose(1, 2)  # [B,T,D]
            toks = toks + self.vision_pos[:, :toks.shape[1]]
            return self.vision_tf(toks)
        except Exception:
            b = screen.shape[0]
            return screen.new_zeros((b, 1, self.config.D_MODEL))

    def forward(self, screen: torch.Tensor, semantic: torch.Tensor,
                prev_cause: Optional[torch.Tensor] = None) -> torch.Tensor:
        screen_feat = self.screen_encoder(screen)
        semantic_feat = self.semantic_encoder(semantic)

        combined = torch.cat([screen_feat, semantic_feat], dim=-1)
        base = self.fusion(combined)
        # mezcla por compuerta: w_vis * vis' + w_sem * sem' + base residual
        gate = self.fusion_gate(combined)  # [B,2]
        w_vis = gate[..., 0:1]
        w_sem = gate[..., 1:2]
        out = base + w_vis * self.visual_bias(screen_feat) \
            + w_sem * self.semantic_bias(semantic_feat)

        # --- razonamiento multimodal: tokens visuales <-> semantico ---
        try:
            vis_toks = self.vision_reason(screen)  # [B,T,D]
            sem_tok = semantic_feat.unsqueeze(1)  # [B,1,D]
            # semantico interroga a la vision: "que ve relevante?"
            sem_ctx, w1 = self.cross_sem2vis(
                sem_tok, vis_toks, vis_toks, need_weights=True)
            # vision interroga al semantico: "que contexto tengo?"
            vis_ctx, _ = self.cross_vis2sem(
                vis_toks, sem_tok, sem_tok, need_weights=False)
            self._last_vis_attn = w1.detach()  # [B,1,T] para focus_map
            reasoned = self.reason_norm(
                sem_ctx.squeeze(1) + vis_ctx.mean(dim=1))
            out = out + self.reason_mlp(reasoned)
        except Exception:
            pass
        # --- tercer modo: causa del paso anterior condiciona el estado ---
        try:
            if prev_cause is not None:
                pc = prev_cause.to(out.device, dtype=out.dtype)
                if pc.dim() == 1:
                    pc = pc.unsqueeze(0).expand(out.shape[0], -1)
                out = out + self.cause_embed(pc)
        except Exception:
            pass
        return out

    def visual_recon_loss(self, screen: torch.Tensor) -> torch.Tensor:
        """MSE entre decoder(screen_feat) y pantalla gris 30x28. Auxiliar."""
        try:
            screen_feat = self.screen_encoder(screen)
            pred = self.vis_decoder(screen_feat).view(
                screen.shape[0], 1, self.FOCUS_H, 28)
            with torch.no_grad():
                gray = screen.mean(dim=1, keepdim=True)  # [B,1,240,256]
                target = F.adaptive_avg_pool2d(gray, (self.FOCUS_H, 28))
            return F.mse_loss(pred, target)
        except Exception:
            return screen.new_zeros(())


# ============================================================================
# ACTION DECODER
# ============================================================================

class ActionDecoder(nn.Module):
    """Decodes model output into action probabilities.

    ``cause`` es la cabeza auxiliar del PORQUE: predice que causas de reward
    (16, multi-etiqueta) dispararon ESTE estado. Obliga a la representacion
    a separar "moneda" de "avanza" de "quieto" en vez de un escalar opaco.
    """

    N_CAUSES = 16

    def __init__(self, config: TopoMarioConfig):
        super().__init__()
        self.config = config

        self.trunk = nn.Sequential(
            nn.Linear(config.D_MODEL, config.D_MODEL),
            nn.ReLU(),
        )
        self.policy = nn.Linear(config.D_MODEL, config.N_ACTIONS)
        self.value = nn.Linear(config.D_MODEL, 1)
        self.cause = nn.Linear(config.D_MODEL, self.N_CAUSES)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.policy(h), self.value(h).squeeze(-1)

    def forward_full(self, x: torch.Tensor):
        h = self.trunk(x)
        return self.policy(h), self.value(h).squeeze(-1), self.cause(h)


# ============================================================================
# TOPOMARIO MODEL
# ============================================================================

class TopoMario(nn.Module):
    """TopoMario: Complex-valued spectral model for playing Super Mario Bros."""

    def __init__(self, config: TopoMarioConfig):
        super().__init__()
        self.config = config

        self.state_encoder = GameStateEncoder(config)

        # tiempo liquido + recuerdo episodico sobre el estado fusionado,
        # ANTES del transformer (que con S=1 no tiene memoria propia)
        self.liquid = LiquidNeuron(config.D_MODEL)
        self.episodic = EpisodicMemoryCell(config.D_MODEL)

        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.N_LAYERS)
        ])

        self.norm = nn.LayerNorm(config.D_MODEL)
        self.action_decoder = ActionDecoder(config)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Only the spectral quaternion kernels get a small init (handled in their
        # own __init__). Everything else keeps PyTorch's fan-in scaled default so
        # the action logits are not crushed into a constant at init time.
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                bound = 1.0 / max(1.0, module.weight.shape[-1]) ** 0.5
                nn.init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, screen: torch.Tensor, semantic: torch.Tensor,
                prev_cause: Optional[torch.Tensor] = None,
                return_value: bool = True, return_cause: bool = False):
        x = self.state_encoder(screen, semantic, prev_cause)
        # neuronas liquidas + memoria episodica: el estado recuerda
        # correlaciones recientes (Hebb local, sin BPTT -> apto para PPO)
        try:
            x = self.liquid(x, plasticity=self.config.LIQUID_PLASTICITY)
            x = self.episodic(x, plasticity=self.config.EPISODIC_PLASTICITY)
        except Exception:
            pass
        x = x.unsqueeze(1)

        total_recon_loss = 0.0
        use_ckpt = self.config.GRADIENT_CHECKPOINTING and self.training
        for layer in self.layers:
            if use_ckpt:
                # recomputa activaciones en backward: -30/40% VRAM en xl/xxl
                # a cambio de +20% tiempo. Solo en train (en act/eval directo).
                x, recon_loss = grad_ckpt(layer, x, use_reentrant=False)
            else:
                x, recon_loss = layer(x)
            total_recon_loss += recon_loss
        # aux visual: que la vision conserve geometria (idea neuromario decoder)
        try:
            total_recon_loss = total_recon_loss + \
                self.config.AE_RECON_WEIGHT * self.state_encoder.visual_recon_loss(screen)
        except Exception:
            pass

        x = self.norm(x)
        feat = x.squeeze(1)
        action_logits, value, cause_logits = self.action_decoder.forward_full(feat)

        if return_cause:
            if return_value:
                return action_logits, value, total_recon_loss, cause_logits
            return action_logits, total_recon_loss, cause_logits
        if return_value:
            return action_logits, value, total_recon_loss
        return action_logits, total_recon_loss

    def get_action(self, screen: torch.Tensor, semantic: torch.Tensor) -> int:
        """Get action for given game state."""
        self.eval()
        with torch.no_grad():
            logits, _, _ = self.forward(screen, semantic)
            probs = F.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1).item()
        return action

    def save_checkpoint(self, path: str, optimizer=None, epoch: int = 0, step: int = 0):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        state = {
            'model': self.state_dict(),
            'config': asdict(self.config),
            'epoch': epoch,
            'step': step,
        }

        if optimizer is not None:
            state['optimizer'] = optimizer.state_dict()

        st_save(state['model'], path)

        meta_path = path.replace('.safetensors', '_meta.json')
        with open(meta_path, 'w') as f:
            json.dump({k: v for k, v in state.items() if k != 'model'}, f, indent=2)

    def load_checkpoint(self, path: str, optimizer=None):
        """Load model checkpoint."""
        state = st_load(path)

        self.load_state_dict(state, strict=False)

        meta_path = path.replace('.safetensors', '_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            if optimizer is not None and 'optimizer' in meta:
                optimizer.load_state_dict(meta['optimizer'])
            return meta
        return {}

"""StatePlay DiT: Wan2.2-TI2V-5B with action and game-state conditioning.

The standalone inference implementation preserves the StatePlay checkpoint key
layout while omitting training-only gradient and sequence-parallel machinery.
"""
import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


try:
    import flash_attn
    _FLASH_ATTN_2 = True
except ImportError:
    _FLASH_ATTN_2 = False


def _attention(q, k, v, num_heads):
    """Multi-head scaled-dot-product attention. Uses Flash-Attn 2 if installed."""
    if _FLASH_ATTN_2 and q.is_cuda:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        return rearrange(x, "b s n d -> b s (n d)")
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    x = F.scaled_dot_product_attention(q, k, v)
    return rearrange(x, "b n s d -> b s (n d)")


def _modulate(x, shift, scale):
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim, position):
    sin = torch.outer(
        position.type(torch.float64),
        torch.pow(10000, -torch.arange(dim // 2, dtype=torch.float64,
                                       device=position.device).div(dim // 2))
    )
    return torch.cat([torch.cos(sin), torch.sin(sin)], dim=1).to(position.dtype)


def _precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].double() / dim))
    freqs = torch.outer(torch.arange(end), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def _precompute_freqs_cis_3d(dim: int, end: int = 1024):
    return (
        _precompute_freqs_cis(dim - 2 * (dim // 3), end),
        _precompute_freqs_cis(dim // 3, end),
        _precompute_freqs_cis(dim // 3, end),
    )


def _rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_c = torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:-1], -1, 2))
    x_out = torch.view_as_real(x_c * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y.to(dtype) * self.weight


class _SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = _rope_apply(q, freqs, self.num_heads)
        k = _rope_apply(k, freqs, self.num_heads)
        return self.o(_attention(q, k, v, self.num_heads))


class _CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(self, x, ctx):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        return self.o(_attention(q, k, v, self.num_heads))


class _BottleneckSelfAttention(nn.Module):
    def __init__(self, dim, inner_dim=512, num_heads=8, eps=1e-6):
        super().__init__()
        if inner_dim % num_heads != 0:
            raise ValueError(f"inner_dim={inner_dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.q = nn.Linear(dim, inner_dim)
        self.k = nn.Linear(dim, inner_dim)
        self.v = nn.Linear(dim, inner_dim)
        self.o = nn.Linear(inner_dim, dim)
        self.norm_q = RMSNorm(inner_dim, eps=eps)
        self.norm_k = RMSNorm(inner_dim, eps=eps)

    def forward(self, x, freqs=None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        return self.o(_attention(q, k, v, self.num_heads))


class _BottleneckCrossAttention(nn.Module):
    def __init__(self, dim, inner_dim=512, num_heads=8, eps=1e-6):
        super().__init__()
        if inner_dim % num_heads != 0:
            raise ValueError(f"inner_dim={inner_dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.q = nn.Linear(dim, inner_dim)
        self.k = nn.Linear(dim, inner_dim)
        self.v = nn.Linear(dim, inner_dim)
        self.o = nn.Linear(inner_dim, dim)
        self.norm_q = RMSNorm(inner_dim, eps=eps)
        self.norm_k = RMSNorm(inner_dim, eps=eps)

    def forward(self, x, y):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(y))
        v = self.v(y)
        return self.o(_attention(q, k, v, self.num_heads))


class _BottleneckFFN(nn.Module):
    def __init__(self, dim, inner_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(inner_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class _Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x, t_mod):
        # t_mod: [B, C] (separated=False) or [B, T, C]
        if t_mod.dim() == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype,
                                                            device=t_mod.device)
                            + t_mod.unsqueeze(2)).chunk(2, dim=2)
            return self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2))
        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
                        + t_mod).chunk(2, dim=1)
        return self.head(self.norm(x) * (1 + scale) + shift)


class _StatePlayActionBackbone(nn.Module):
    """Wan2.2-TI2V-5B DiT with 30 per-block action embedders.

    Action conditioning: each `nn.Linear(num_buttons, dim)` projects the
    temporally binned keyboard signal and adds the result before its DiTBlock.
    Action embeddings are injected directly, without gates.
    """
    def __init__(
        self,
        num_buttons: int = 10,
        dim: int = 3072,
        in_dim: int = 48,
        ffn_dim: int = 14336,
        out_dim: int = 48,
        text_dim: int = 4096,
        freq_dim: int = 256,
        num_heads: int = 24,
        num_layers: int = 30,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_buttons = num_buttons

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            _StatePlayBlock(dim, num_heads, ffn_dim, eps) for _ in range(num_layers)
        ])
        self.head = _Head(dim, out_dim, patch_size, eps)
        self.action_embedders = nn.ModuleList([
            nn.Linear(num_buttons, dim, bias=False) for _ in range(num_layers)
        ])

        head_dim = dim // num_heads
        self.freqs = _precompute_freqs_cis_3d(head_dim)

    def _bin_action(self, keyboard: Optional[torch.Tensor], f: int):
        if keyboard is None:
            return None
        # keyboard: [B, num_raw, K] dense. Adaptive max-pool → [B, f, K].
        return F.adaptive_max_pool1d(keyboard.transpose(1, 2), output_size=f).transpose(1, 2)


class _StatePlayStateBackbone(_StatePlayActionBackbone):
    """State-token backbone: jointly denoise video latents and 5D game-state tokens."""

    def __init__(self, num_buttons: int = 10, state_dim: int = 5, **kwargs):
        super().__init__(num_buttons=num_buttons, **kwargs)
        self.state_dim = state_dim
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, 256),
            nn.GELU(),
            nn.Linear(256, self.dim),
            nn.LayerNorm(self.dim),
        )
        self.state_decoder = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, 256),
            nn.GELU(),
            nn.Linear(256, state_dim),
        )

    def _inject_action_state(self, x, action_binned, block_id, f, h, w):
        if action_binned is None:
            return x
        B = action_binned.shape[0]
        video_len = f * h * w
        state_len = f
        if x.shape[1] != video_len + state_len:
            raise AssertionError(
                f"state-aware token length mismatch: got {x.shape[1]}, "
                f"expected {video_len + state_len}"
            )
        emb = self.action_embedders[block_id](action_binned.to(x.dtype))
        video_bias = emb.unsqueeze(2).expand(-1, -1, h * w, -1).reshape(B, video_len, self.dim)
        state_bias = emb
        return x + torch.cat([video_bias, state_bias], dim=1)


class _StatePlayBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, eps=1e-6):
        super().__init__()
        self.self_attn = _SelfAttention(dim, num_heads, eps)
        self.cross_attn = _CrossAttention(dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim),
        )

        self.state_self_attn = _BottleneckSelfAttention(dim, inner_dim=512, num_heads=8, eps=eps)
        self.state_norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.state_norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.state_ffn = _BottleneckFFN(dim, inner_dim=1024)

        self.video_to_state_attn = _BottleneckCrossAttention(dim, inner_dim=512, num_heads=8, eps=eps)
        self.state_to_video_attn = _BottleneckCrossAttention(dim, inner_dim=512, num_heads=8, eps=eps)
        self.norm_video_to_state_q = nn.LayerNorm(dim, eps=eps)
        self.norm_video_to_state_kv = nn.LayerNorm(dim, eps=eps)
        self.norm_state_to_video_q = nn.LayerNorm(dim, eps=eps)
        self.norm_state_to_video_kv = nn.LayerNorm(dim, eps=eps)

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def _mods(self, t_mod):
        has_seq = t_mod.dim() == 4
        chunk_dim = 2 if has_seq else 1
        mods = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            mods = [m.squeeze(2) for m in mods]
        return mods

    def forward(self, video, state, context, t_mod_video, t_mod_state, video_freqs, state_freqs):
        shift_msa_v, scale_msa_v, gate_msa_v, shift_mlp_v, scale_mlp_v, gate_mlp_v = self._mods(t_mod_video)
        shift_msa_s, scale_msa_s, gate_msa_s, shift_mlp_s, scale_mlp_s, gate_mlp_s = self._mods(t_mod_state)

        video = video + gate_msa_v * self.self_attn(_modulate(self.norm1(video), shift_msa_v, scale_msa_v), video_freqs)
        state = state + gate_msa_s * self.state_self_attn(_modulate(self.state_norm1(state), shift_msa_s, scale_msa_s), state_freqs)

        video = video + self.cross_attn(self.norm3(video), context)

        video_kv = video
        state_kv = state
        video = video + self.video_to_state_attn(
            self.norm_video_to_state_q(video),
            self.norm_video_to_state_kv(state_kv),
        )
        state = state + self.state_to_video_attn(
            self.norm_state_to_video_q(state),
            self.norm_state_to_video_kv(video_kv),
        )

        video = video + gate_mlp_v * self.ffn(_modulate(self.norm2(video), shift_mlp_v, scale_mlp_v))
        state = state + gate_mlp_s * self.state_ffn(_modulate(self.state_norm2(state), shift_mlp_s, scale_mlp_s))
        return video, state


class StatePlayDiT(_StatePlayStateBackbone):
    """StatePlay DiT: split video/state blocks with lightweight state and mutual-attn branches."""

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        keyboard_action: Optional[torch.Tensor] = None,
        noisy_state: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = True,
    ):
        if noisy_state is None:
            raise ValueError("StatePlayDiT requires noisy_state.")
        if noisy_state.ndim != 3 or noisy_state.shape[-1] != self.state_dim:
            raise ValueError(
                f"noisy_state must be [B, F, {self.state_dim}], got {tuple(noisy_state.shape)}"
            )

        ctx = self.text_embedding(context)
        x = latents
        if x.shape[0] != ctx.shape[0]:
            x = torch.cat([x] * ctx.shape[0], dim=0)
        if noisy_state.shape[0] != x.shape[0]:
            noisy_state = torch.cat([noisy_state] * x.shape[0], dim=0)

        x = self.patch_embedding(x)
        f, h, w = x.shape[2:]
        video_tokens = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        B, video_len, C = video_tokens.shape
        tokens_per_frame = h * w

        noisy_state = noisy_state.to(device=latents.device, dtype=latents.dtype)
        if noisy_state.shape[1] != f:
            raise AssertionError(
                f"noisy_state temporal length must match latent f={f}; got {tuple(noisy_state.shape)}"
            )
        state_tokens = self.state_encoder(noisy_state)
        if state_tokens.shape != (B, f, C):
            raise AssertionError(
                f"state token shape mismatch: got {tuple(state_tokens.shape)}, expected {(B, f, C)}"
            )

        if fuse_vae_embedding_in_latents:
            video_ts = torch.cat([
                torch.zeros((tokens_per_frame,), dtype=latents.dtype, device=latents.device),
                torch.ones(((f - 1) * tokens_per_frame,), dtype=latents.dtype, device=latents.device) * timestep,
            ])
            state_ts = torch.cat([
                torch.zeros((1,), dtype=latents.dtype, device=latents.device),
                torch.ones((f - 1,), dtype=latents.dtype, device=latents.device) * timestep,
            ])
            full_ts = torch.cat([video_ts, state_ts], dim=0)
            t_all = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, full_ts).unsqueeze(0))
            t_mod = self.time_projection(t_all).unflatten(2, (6, self.dim))
            t_mod_video = t_mod[:, :video_len]
            t_mod_state = t_mod[:, video_len:video_len + f]
            t_video = t_all[:, :video_len]
        else:
            t_video = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
            t_mod = self.time_projection(t_video).unflatten(1, (6, self.dim))
            t_mod_video = t_mod
            t_mod_state = t_mod

        video_freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(f * h * w, 1, -1).to(video_tokens.device)
        state_freqs = torch.cat([
            self.freqs[0][:f],
            self.freqs[1][:1].expand(f, -1),
            self.freqs[2][:1].expand(f, -1),
        ], dim=-1).reshape(f, 1, -1).to(video_tokens.device)

        action_binned = self._bin_action(keyboard_action, f)
        for i, block in enumerate(self.blocks):
            x_all = torch.cat([video_tokens, state_tokens], dim=1)
            x_all = self._inject_action_state(x_all, action_binned, i, f, h, w)
            video_tokens = x_all[:, :video_len]
            state_tokens = x_all[:, video_len:video_len + f]
            video_tokens, state_tokens = block(
                video_tokens, state_tokens, ctx, t_mod_video, t_mod_state, video_freqs, state_freqs,
            )

        video_out = self.head(video_tokens, t_video)
        video_out = rearrange(
            video_out, "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=f, h=h, w=w,
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2],
        )
        state_out = self.state_decoder(state_tokens)
        return video_out, state_out

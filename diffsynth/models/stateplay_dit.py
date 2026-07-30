import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
from typing import Tuple, Optional
from einops import rearrange, repeat
from .wan_video_camera_controller import SimpleAdapter
from ..core.gradient import gradient_checkpoint_forward

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x,tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    freqs = freqs.to(torch.complex64) if freqs.device.type == "npu" else freqs
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class MutualCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(y))
        v = self.v(y)
        x = self.attn(q, k, v)
        return self.o(x)


class BottleneckSelfAttention(nn.Module):
    """Small state-token self-attention with Wan-dim input/output."""

    def __init__(
        self,
        dim: int,
        inner_dim: int = 512,
        num_heads: int = 8,
        eps: float = 1e-6,
    ):
        super().__init__()
        if inner_dim % num_heads != 0:
            raise ValueError(f"inner_dim={inner_dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.q = nn.Linear(dim, inner_dim)
        self.k = nn.Linear(dim, inner_dim)
        self.v = nn.Linear(dim, inner_dim)
        self.o = nn.Linear(inner_dim, dim)
        self.norm_q = RMSNorm(inner_dim, eps=eps)
        self.norm_k = RMSNorm(inner_dim, eps=eps)
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs=None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        x = self.attn(q, k, v)
        return self.o(x)


class BottleneckCrossAttention(nn.Module):
    """Small cross-attention with Wan-dim input/output."""

    def __init__(
        self,
        dim: int,
        inner_dim: int = 512,
        num_heads: int = 8,
        eps: float = 1e-6,
    ):
        super().__init__()
        if inner_dim % num_heads != 0:
            raise ValueError(f"inner_dim={inner_dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.q = nn.Linear(dim, inner_dim)
        self.k = nn.Linear(dim, inner_dim)
        self.v = nn.Linear(dim, inner_dim)
        self.o = nn.Linear(inner_dim, dim)
        self.norm_q = RMSNorm(inner_dim, eps=eps)
        self.norm_k = RMSNorm(inner_dim, eps=eps)
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(y))
        v = self.v(y)
        x = self.attn(q, k, v)
        return self.o(x)


class BottleneckFFN(nn.Module):
    """Small state-token FFN with Wan-dim input/output."""

    def __init__(self, dim: int, inner_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(inner_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual


class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs, cross_attn_video_len: Optional[int] = None):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
        if cross_attn_video_len is None:
            x = x + self.cross_attn(self.norm3(x), context)
        else:
            if cross_attn_video_len < 0 or cross_attn_video_len > x.shape[1]:
                raise AssertionError(
                    f"cross_attn_video_len={cross_attn_video_len} is invalid for sequence length {x.shape[1]}"
                )
            x_video = x[:, :cross_attn_video_len]
            x_state = x[:, cross_attn_video_len:]
            x_video = x_video + self.cross_attn(self.norm3(x_video), context)
            x = torch.cat([x_video, x_state], dim=1)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class StateVideoDiTBlock(nn.Module):
    """StatePlay block: separate video/state self-attn + FFN with mutual cross-attn."""

    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        # Video branch keeps the original Wan-compatible parameter names so base
        # weights load into the video path unchanged.
        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))

        self.state_self_attn = BottleneckSelfAttention(
            dim, inner_dim=512, num_heads=8, eps=eps,
        )
        self.state_norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.state_norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.state_ffn = BottleneckFFN(dim, inner_dim=1024)

        self.video_to_state_attn = BottleneckCrossAttention(
            dim, inner_dim=512, num_heads=8, eps=eps,
        )
        self.state_to_video_attn = BottleneckCrossAttention(
            dim, inner_dim=512, num_heads=8, eps=eps,
        )
        self.norm_video_to_state_q = nn.LayerNorm(dim, eps=eps)
        self.norm_video_to_state_kv = nn.LayerNorm(dim, eps=eps)
        self.norm_state_to_video_q = nn.LayerNorm(dim, eps=eps)
        self.norm_state_to_video_kv = nn.LayerNorm(dim, eps=eps)

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def _modulation_chunks(self, t_mod):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        chunks = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            chunks = tuple(chunk.squeeze(2) for chunk in chunks)
        return chunks

    def forward(self, video, state, context, t_mod_video, t_mod_state, video_freqs, state_freqs):
        shift_msa_v, scale_msa_v, gate_msa_v, shift_mlp_v, scale_mlp_v, gate_mlp_v = (
            self._modulation_chunks(t_mod_video)
        )
        shift_msa_s, scale_msa_s, gate_msa_s, shift_mlp_s, scale_mlp_s, gate_mlp_s = (
            self._modulation_chunks(t_mod_state)
        )

        input_video = modulate(self.norm1(video), shift_msa_v, scale_msa_v)
        video = self.gate(video, gate_msa_v, self.self_attn(input_video, video_freqs))

        input_state = modulate(self.state_norm1(state), shift_msa_s, scale_msa_s)
        state = self.gate(state, gate_msa_s, self.state_self_attn(input_state, state_freqs))

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

        input_video = modulate(self.norm2(video), shift_mlp_v, scale_mlp_v)
        video = self.gate(video, gate_mlp_v, self.ffn(input_video))

        input_state = modulate(self.state_norm2(state), shift_mlp_s, scale_mlp_s)
        state = self.gate(state, gate_mlp_s, self.state_ffn(input_state))
        return video, state


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.control_adapter = None

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2],
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        if self.has_image_input:
            x = torch.cat([x, y], dim=1)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        x, (f, h, w) = self.patchify(x)

        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        for block in self.blocks:
            if self.training:
                x = gradient_checkpoint_forward(
                    block,
                    use_gradient_checkpointing,
                    use_gradient_checkpointing_offload,
                    x, context, t_mod, freqs
                )
            else:
                x = block(x, context, t_mod, freqs)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x


class _StatePlayActionBackbone(WanModel):
    """
    Wan backbone with per-block independent action embedders.

        30 independent `nn.Linear(num_buttons, dim)` modules (one per DiTBlock)
    each project the downsampled keyboard action and add the result to the
    hidden state right before their block. No LayerScale gate.

    Action-conditioning design:
        Initial attempt A — embedder=0, gates=0      → dead gradient (both ends)
        Initial attempt B — embedder=xavier, gates=0 → gate grew but ∂loss/∂embedder was
                                          attenuated 50× by small gate → bf16
                                          round-to-zero → embedder stuck
        Initial attempt C — embedder=xavier, gates=0.1 → gate=0.1 hit bf16 precision wall
                                          (0.1/128 ≈ 7.8e-4 ≫ lr·grad ≈ 5e-5)
                                          → action params bit-exact frozen
        StatePlay — 30 independent embedders, no gate:
                    ∂loss/∂embedder_i = upstream_grad[i] (no 50× gate attn).
                    Xavier std ≈ 0.025 is the same magnitude known to be
                    bf16-learnable. Each embedder has a shorter gradient path
                    (block 29 backprops through 1 layer; block 0 through 30)
                    and learns its own projection direction, giving the model
                    strictly more representational capacity than a shared emb.

    Trade-off: NOT bit-identical at init. Expected first-step perturbation is
    roughly sqrt(30) · xavier_std · |keyboard_action| ≈ 17% of the hidden-state
    scale per channel. We accept this because StatePlay training start from the raw
    Wan2.2-TI2V-5B backbone (no MG3 visual warm-start), so there is no fragile
    baseline to preserve.

    Parameter count: 30 × (num_buttons × dim) = 30 × 10 × 3072 ≈ 920K
    (0.018% of the 5B backbone; negligible).

    Two methods:
      - `prepare_action_binned(keyboard_action, f)` → `[B, f, num_buttons]`
        Pure temporal downsample. Call ONCE per forward, before the block loop.
      - `inject_at_block(x, action_binned, block_id, f, h, w)` → `[B, L, C]`
        Called by `diffsynth.pipelines.stateplay.model_fn_stateplay` before
        each DiTBlock.
    """
    def __init__(self, num_buttons: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.num_buttons = num_buttons
        self.action_embedders = nn.ModuleList([
            nn.Linear(num_buttons, self.dim, bias=False)
            for _ in range(len(self.blocks))
        ])
        for emb in self.action_embedders:
            nn.init.xavier_normal_(emb.weight)

    def prepare_action_binned(self, keyboard_action, f):
        """
        Downsample raw keyboard action to latent frame rate via adaptive max-pool.

        Called ONCE per forward; output is shared across all 30 blocks (each
        block runs it through its own embedder in `inject_at_block`).

        keyboard_action: [B, num_raw, num_buttons]  dense 0/1 (assumes hold-last
                                                    upsample already applied)
        f:               latent temporal dim (StatePlay 101-frame: f=26)

        Returns: [B, f, num_buttons] binned action, or None if input was None.
        """
        if keyboard_action is None:
            return None
        # "Did this button fire anywhere in the bin?" Output stays in {0, 1}.
        return F.adaptive_max_pool1d(
            keyboard_action.transpose(1, 2),    # [B, num_buttons, num_raw]
            output_size=f,
        ).transpose(1, 2)                        # [B, f, num_buttons]

    def inject_at_block(self, x, action_binned, block_id, f, h, w):
        """
        Project action_binned with this block's own embedder and add to x.

        x:              [B, L, C] hidden state before DiTBlock `block_id`.
                        L is usually f*h*w. When reference_latents is prepended
                        upstream, L = f*h*w + prefix_len; we zero-pad the bias
                        for the prefix tokens so they are untouched.
        action_binned:  [B, f, num_buttons] from prepare_action_binned, or None
        block_id:       int in [0, num_layers)
        f, h, w:        latent spatiotemporal dims captured BEFORE any
                        reference_latents concat (so f*h*w matches the emb
                        output length regardless of downstream prefixes)
        """
        if action_binned is None:
            return x
        B = action_binned.shape[0]
        C = self.dim
        action_emb = self.action_embedders[block_id](action_binned.to(x.dtype))  # [B, f, C]
        action_bias = (action_emb
                       .unsqueeze(2)                       # [B, f, 1, C]
                       .expand(-1, -1, h * w, -1)          # [B, f, h*w, C] (stride view)
                       .reshape(B, f * h * w, C))          # [B, f*h*w, C]
        if x.shape[1] == action_bias.shape[1]:
            return x + action_bias
        # x has a reference_latents prefix; pad the bias with zeros for it.
        prefix_len = x.shape[1] - action_bias.shape[1]
        padding = torch.zeros(B, prefix_len, C, dtype=x.dtype, device=x.device)
        padded_bias = torch.cat([padding, action_bias], dim=1)
        return x + padded_bias


class _StatePlayStateBackbone(_StatePlayActionBackbone):
    """StatePlay state backbone: joint video/state-token DiT with per-block actions."""

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
        self._state_shape_logged = False

    def encode_state(self, noisy_state: torch.Tensor) -> torch.Tensor:
        if noisy_state is None:
            raise ValueError("_StatePlayStateBackbone requires state/noisy_state input.")
        if noisy_state.shape[-1] != self.state_dim:
            raise AssertionError(
                f"state.shape[-1] must be {self.state_dim}; got {tuple(noisy_state.shape)}"
            )
        return self.state_encoder(noisy_state)

    def decode_state(self, state_tokens: torch.Tensor) -> torch.Tensor:
        return self.state_decoder(state_tokens)

    def inject_at_block_state(self, x, action_binned, block_id, f, h, w):
        """Add action bias to concatenated [video_tokens, state_tokens]."""
        if action_binned is None:
            return x
        B = action_binned.shape[0]
        C = self.dim
        Lv = f * h * w
        Ls = f
        if x.shape[1] != Lv + Ls:
            raise AssertionError(
                f"state-aware sequence length mismatch: got {x.shape[1]}, expected {Lv + Ls}"
            )
        action_emb = self.action_embedders[block_id](action_binned.to(x.dtype))
        video_bias = (action_emb
                      .unsqueeze(2)
                      .expand(-1, -1, h * w, -1)
                      .reshape(B, Lv, C))
        state_bias = action_emb
        return x + torch.cat([video_bias, state_bias], dim=1)


class StatePlayModel(_StatePlayStateBackbone):
    """StatePlay: clean-state loss with split video/state blocks."""

    def __init__(self, num_buttons: int = 10, state_dim: int = 5, **kwargs):
        super().__init__(num_buttons=num_buttons, state_dim=state_dim, **kwargs)
        num_layers = len(self.blocks)
        ffn_dim = self.blocks[0].ffn_dim
        eps = self.blocks[0].norm1.eps
        num_heads = self.blocks[0].num_heads
        self.blocks = nn.ModuleList([
            StateVideoDiTBlock(self.has_image_input, self.dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])

    def init_state_branch_from_pretrained(self, pretrained_state):
        own_state = self.state_dict()
        mapped = {}
        for block_id in range(len(self.blocks)):
            for dst_prefix, src_prefix in (
                (f"blocks.{block_id}.state_self_attn.", f"blocks.{block_id}.self_attn."),
                (f"blocks.{block_id}.state_ffn.", f"blocks.{block_id}.ffn."),
            ):
                for dst_name in own_state:
                    if not dst_name.startswith(dst_prefix):
                        continue
                    src_name = src_prefix + dst_name[len(dst_prefix):]
                    src = pretrained_state.get(src_name)
                    if src is not None and src.shape == own_state[dst_name].shape:
                        mapped[dst_name] = src
        if mapped:
            self.load_state_dict(mapped, strict=False)
        return len(mapped)


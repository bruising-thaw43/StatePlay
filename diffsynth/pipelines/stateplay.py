"""StatePlay training pipeline — Wan2.2-TI2V-5B + per-block action conditioning.

Slimmed-down fork of the Wan video pipeline that keeps only what the
StatePlay training needs:

  • Single DiT (`pipe.dit`)              — Wan2.2-TI2V-5B is monolithic, no MoE.
  • T5 prompt encoder + tokenizer.
  • Wan2.2 VAE.
  • 5 pipeline units                     — ShapeChecker, NoiseInitializer,
                                           PromptEmbedder, InputVideoEmbedder,
                                           ImageEmbedderFused.
  • `model_fn_stateplay` — joint video/state StatePlay forward.

Only the model components and preprocessing units required by StatePlay training are included.
"""
import torch
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Union

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from ..models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from ..models.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
from ..models.wan_video_vae import WanVideoVAE


class StatePlayTrainingPipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
            time_division_factor=4, time_division_remainder=1,
        )
        self.scheduler = FlowMatchScheduler("Wan")
        self.tokenizer: HuggingfaceTokenizer = None
        self.text_encoder: WanTextEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.in_iteration_models = ("dit",)
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_ImageEmbedderFused(),
        ]
        self.model_fn = model_fn_stateplay


    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="google/umt5-xxl/",
        ),
        redirect_common_files: bool = True,
        vram_limit: float = None,
    ):
        # Redirect common .pth filenames to converted safetensors mirrors.
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": (
                    "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
                    "models_t5_umt5-xxl-enc-bf16.safetensors",
                ),
                "Wan2.2_VAE.pth": (
                    "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
                    "Wan2.2_VAE.safetensors",
                ),
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if (model_config.origin_file_pattern in redirect_dict
                        and model_config.model_id != redirect_dict[model_config.origin_file_pattern][0]):
                    print(
                        f"To avoid repeatedly downloading model files, "
                        f"({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to "
                        f"{redirect_dict[model_config.origin_file_pattern]}. "
                        f"You can use `redirect_common_files=False` to disable file redirection."
                    )
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern][0]
                    model_config.origin_file_pattern = redirect_dict[model_config.origin_file_pattern][1]

        pipe = StatePlayTrainingPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        # Fetch the three models StatePlay needs. The DiT loader accepts
        # index=2 (Wan2.x convention); for Wan2.2-TI2V-5B it returns a single
        # WanModel — if a list slips through (mis-configured Wan2.2 MoE), we
        # train against the first expert and warn the user.
        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
        dit = model_pool.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            print(f"[StatePlayTrainingPipeline] DiT loader returned {len(dit)} experts; "
                  "StatePlay trains a single DiT, taking expert 0.")
            pipe.dit = dit[0]
        else:
            pipe.dit = dit
        pipe.vae = model_pool.fetch_model("wan_video_vae")

        # Size division factor: VAE downsamples 8x but DiT also patches by 2x,
        # so the H/W must be divisible by 16 (Wan2.2 VAE upsampling_factor=16).
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = HuggingfaceTokenizer(
                name=tokenizer_config.path, seq_len=512, clean='whitespace',
            )

        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe


class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "num_frames"),
        )

    def process(self, pipe: StatePlayTrainingPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}



class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "seed", "rand_device"),
            output_params=("noise",),
        )

    def process(self, pipe: StatePlayTrainingPipeline, height, width, num_frames, seed, rand_device):
        length = (num_frames - 1) // 4 + 1
        shape = (
            1, pipe.vae.model.z_dim, length,
            height // pipe.vae.upsampling_factor,
            width // pipe.vae.upsampling_factor,
        )
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        return {"noise": noise}



class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",),
        )

    def process(self, pipe: StatePlayTrainingPipeline, input_video, noise, tiled, tile_size, tile_stride):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(self.onload_model_names)
        input_video = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode(
            input_video, device=pipe.device,
            tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        if pipe.scheduler.training:
            # Training feeds noise + input_latents to FlowMatchSFTLoss separately;
            # the loss adds noise via the scheduler, so latents starts as noise.
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}



class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            output_params=("context",),
            onload_model_names=("text_encoder",),
        )

    def encode_prompt(self, pipe: StatePlayTrainingPipeline, prompt):
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def process(self, pipe: StatePlayTrainingPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = self.encode_prompt(pipe, prompt)
        return {"context": prompt_emb}



class WanVideoUnit_ImageEmbedderFused(PipelineUnit):
    """
    Encode the input first frame to a latent and overwrite latents[:, :, 0:1].

    Specific to Wan2.2-TI2V-5B (`fuse_vae_embedding_in_latents=True`); the I2V
    conditioning rides on the first latent slot rather than as an extra `y`
    channel concat.
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "latents", "height", "width",
                          "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "fuse_vae_embedding_in_latents", "first_frame_latents"),
            onload_model_names=("vae",),
        )

    def process(self, pipe: StatePlayTrainingPipeline, input_image, latents,
                height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.fuse_vae_embedding_in_latents:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).transpose(0, 1)
        z = pipe.vae.encode(
            [image], device=pipe.device,
            tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
        )
        latents[:, :, 0: 1] = z
        return {"latents": latents, "fuse_vae_embedding_in_latents": True, "first_frame_latents": z}



def _squeeze_state_batch_dim(state: torch.Tensor) -> torch.Tensor:
    if state is None:
        return state
    if state.ndim == 4 and state.shape[1] == 1:
        state = state[:, 0]
    if state.ndim != 3:
        raise ValueError(
            f"[StatePlay state] expected state shape [B, T, 5] or [B, 1, T, 5], got {tuple(state.shape)}"
        )
    return state


def downsample_state_to_latent(state: torch.Tensor, f: int, sampling: str = "interpolation") -> torch.Tensor:
    state = _squeeze_state_batch_dim(state)
    if state.shape[-1] != 5:
        raise AssertionError(f"state.shape[-1] == 5 required; got {tuple(state.shape)}")
    if state.shape[1] == f:
        return state
    if sampling == "interpolation":
        return F.interpolate(
            state.transpose(1, 2).float(), size=f, mode="linear", align_corners=True,
        ).transpose(1, 2).to(dtype=state.dtype)
    if sampling == "avg":
        return F.adaptive_avg_pool1d(
            state.transpose(1, 2).float(), output_size=f,
        ).transpose(1, 2).to(dtype=state.dtype)
    if sampling == "end":
        T = state.shape[1]
        # Match PyTorch adaptive pooling bin ends: end_i = ceil((i + 1) * T / f).
        end_idx = torch.ceil(
            torch.arange(1, f + 1, device=state.device, dtype=torch.float32) * T / f
        ).long() - 1
        end_idx = end_idx.clamp(min=0, max=T - 1)
        return state.index_select(1, end_idx)
    raise ValueError(
        f"[StatePlay state] unsupported state_sampling={sampling!r}; "
        "expected one of: interpolation, avg, end"
    )


def model_fn_stateplay(
    dit: WanModel,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    fuse_vae_embedding_in_latents: bool = False,
    keyboard_action: Optional[torch.Tensor] = None,
    noisy_state: Optional[torch.Tensor] = None,
    state: Optional[torch.Tensor] = None,
    **kwargs,
):
    """StatePlay forward: split video/state blocks with mutual cross-attn."""
    if noisy_state is None:
        noisy_state = state
    if noisy_state is None:
        raise ValueError("[StatePlay state] model_fn_stateplay requires noisy_state/state.")
    noisy_state = _squeeze_state_batch_dim(noisy_state).to(device=latents.device, dtype=latents.dtype)
    if noisy_state.shape[-1] != getattr(dit, "state_dim", 5):
        raise AssertionError(
            f"state.shape[-1] == {getattr(dit, 'state_dim', 5)} required; got {tuple(noisy_state.shape)}"
        )

    context = dit.text_embedding(context)

    x = dit.patchify(latents)
    f, h, w = x.shape[2:]
    video_tokens = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
    B, Lv, C = video_tokens.shape
    tokens_per_frame = h * w

    if noisy_state.shape[1] != f:
        raise AssertionError(
            f"noisy_state temporal length must equal latent f={f}; got {tuple(noisy_state.shape)}"
        )
    state_tokens = dit.encode_state(noisy_state)
    if state_tokens.shape != (B, f, C):
        raise AssertionError(
            f"state_tokens shape mismatch: got {tuple(state_tokens.shape)}, expected {(B, f, C)}"
        )

    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        video_timestep = torch.cat([
            torch.zeros((tokens_per_frame,), dtype=latents.dtype, device=latents.device),
            torch.ones(((f - 1) * tokens_per_frame,), dtype=latents.dtype, device=latents.device) * timestep,
        ])
        state_timestep = torch.cat([
            torch.zeros((1,), dtype=latents.dtype, device=latents.device),
            torch.ones((f - 1,), dtype=latents.dtype, device=latents.device) * timestep,
        ])
        full_timestep = torch.cat([video_timestep, state_timestep], dim=0)
        t_all = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, full_timestep).unsqueeze(0))
        t_mod = dit.time_projection(t_all).unflatten(2, (6, dit.dim))
        t_mod_video = t_mod[:, :Lv]
        t_mod_state = t_mod[:, Lv:Lv + f]
        t_video = t_all[:, :Lv]
    else:
        t_video = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t_video).unflatten(1, (6, dit.dim))
        t_mod_video = t_mod
        t_mod_state = t_mod

    video_freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, 1, -1).to(video_tokens.device)
    state_freqs = torch.cat([
        dit.freqs[0][:f],
        dit.freqs[1][:1].expand(f, -1),
        dit.freqs[2][:1].expand(f, -1),
    ], dim=-1).reshape(f, 1, -1).to(video_tokens.device)

    action_binned = None
    if keyboard_action is not None and hasattr(dit, 'prepare_action_binned'):
        action_binned = dit.prepare_action_binned(keyboard_action, f)

    if not getattr(dit, "_state_shape_logged", False):
        print(
            "[StatePlay shapes] "
            f"f={f} h={h} w={w} tokens_per_frame={tokens_per_frame} "
            f"Lv={Lv} Ls={f} state_dim={noisy_state.shape[-1]} "
            f"action_dim={None if keyboard_action is None else keyboard_action.shape[-1]} "
            "text_cross_attn=video_only mutual_cross_attn=video_state"
        )
        dit._state_shape_logged = True

    for block_id, block in enumerate(dit.blocks):
        if action_binned is not None and hasattr(dit, 'inject_at_block_state'):
            x = torch.cat([video_tokens, state_tokens], dim=1)
            x = dit.inject_at_block_state(x, action_binned, block_id, f, h, w)
            video_tokens = x[:, :Lv]
            state_tokens = x[:, Lv:Lv + f]
        video_tokens, state_tokens = gradient_checkpoint_forward(
            block,
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
            video_tokens, state_tokens, context, t_mod_video, t_mod_state, video_freqs, state_freqs,
        )

    pred_video = dit.head(video_tokens, t_video)
    pred_video = dit.unpatchify(pred_video, (f, h, w))
    pred_state = dit.decode_state(state_tokens)
    if pred_state.shape != (B, f, getattr(dit, "state_dim", 5)):
        raise AssertionError(
            f"pred_state shape mismatch: got {tuple(pred_state.shape)}, expected {(B, f, getattr(dit, 'state_dim', 5))}"
        )
    return pred_video, pred_state


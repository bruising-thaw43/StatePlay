"""StatePlay training entry point.

This is the single StatePlay training path with one public model name and no architecture selector.
"""
import argparse
import os
import sys
import warnings
from pathlib import Path

import accelerate
import torch
from safetensors.torch import load_file

# Make the example's `data/` package importable when this script is launched
# directly via `python training/train.py`.
_HERE = Path(__file__).resolve().parent
_STATEPLAY = _HERE.parent
_REPO_ROOT = _STATEPLAY.parent
for _path in (str(_STATEPLAY), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from data.action_utils import get_action_op  # noqa: E402
from data.state_utils import StateAugmentedDataset  # noqa: E402
from data.profiles import SF3  # noqa: E402
from data.prompt_utils import resolve_prompt  # noqa: E402

from diffsynth.core import UnifiedDataset  # noqa: E402
from diffsynth.diffusion import *  # noqa: E402,F401,F403
from diffsynth.models.stateplay_dit import StatePlayModel  # noqa: E402
from diffsynth.pipelines.stateplay import (  # noqa: E402
    ModelConfig,
    StatePlayTrainingPipeline,
    model_fn_stateplay,
    downsample_state_to_latent,
)

# Local launcher that mirrors upstream launch_training_task and adds
# accelerator.save_state / load_state for seamless resume.
from runner_with_state import launch_training_task_with_state  # noqa: E402

os.environ["TOKENIZERS_PARALLELISM"] = "false"

WAN_MODEL_KWARGS = {
    "has_image_input": False,
    "patch_size": [1, 2, 2],
    "in_dim": 48, "dim": 3072, "ffn_dim": 14336, "freq_dim": 256,
    "text_dim": 4096, "out_dim": 48, "num_heads": 24, "num_layers": 30,
    "eps": 1e-06,
    "seperated_timestep": True,
    "require_clip_embedding": False,
    "require_vae_embedding": False,
    "fuse_vae_embedding_in_latents": True,
}



def _state_debug_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _reserve_cuda_memory(gb: float) -> None:
    if gb <= 0 or not torch.cuda.is_available():
        return
    device = torch.device("cuda", torch.cuda.current_device())
    target_bytes = int(gb * (1024 ** 3))
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes <= target_bytes:
        raise RuntimeError(
            f"Cannot reserve {gb:.1f} GiB on {device}: "
            f"only {free_bytes / (1024 ** 3):.1f} GiB free of "
            f"{total_bytes / (1024 ** 3):.1f} GiB total."
        )
    reserve = torch.empty(target_bytes // 2, dtype=torch.float16, device=device)
    reserve.zero_()
    del reserve
    torch.cuda.synchronize(device)
    reserved_bytes = torch.cuda.memory_reserved(device)
    print(
        f"[CUDA reserve] device={device} requested={gb:.1f}GiB "
        f"reserved_by_allocator={reserved_bytes / (1024 ** 3):.1f}GiB",
        flush=True,
    )


def _parse_float_csv(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _denormalize_state_for_debug(state: torch.Tensor, norm_max: str) -> torch.Tensor:
    norm = torch.tensor(_parse_float_csv(norm_max), device=state.device, dtype=state.dtype)
    return ((state.clamp(-1.0, 1.0) + 1.0) * 0.5) * norm


def _append_state_debug_record(
    path: str,
    step: int,
    timestep: torch.Tensor,
    sigma: torch.Tensor,
    pred_clean_state: torch.Tensor,
    true_clean_state: torch.Tensor,
    state_columns: str,
    state_norm_max: str,
):
    columns = [c.strip() for c in state_columns.split(",") if c.strip()]
    pred = _denormalize_state_for_debug(pred_clean_state.detach().float()[0], state_norm_max).cpu()
    true = _denormalize_state_for_debug(true_clean_state.detach().float()[0], state_norm_max).cpu()
    err = (pred - true).abs()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n=== step {step} | timestep={float(timestep.detach().float().cpu().flatten()[0]):.3f} | sigma={float(sigma.detach().float().cpu().flatten()[0]):.6f} ===\n")
        f.write(f"{'frame':>5}")
        for col in columns:
            f.write(f" {col + '_pred':>12} {col + '_true':>12} {col + '_err':>12}")
        f.write("\n")
        for i in range(pred.shape[0]):
            f.write(f"{i:5d}")
            for j in range(len(columns)):
                f.write(
                    f" {pred[i, j].item():12.3f}"
                    f" {true[i, j].item():12.3f}"
                    f" {err[i, j].item():12.3f}"
                )
            f.write("\n")


def FlowMatchSFTLossWithStateWeighted(
    pipe: StatePlayTrainingPipeline,
    state_loss_weight: float = 1.0,
    video_loss_weight: float = 1.0,
    state_sampling: str = "interpolation",
    state_debug_enabled: bool = False,
    state_debug_step: int | None = None,
    state_debug_interval: int = 1000,
    state_debug_path: str | None = None,
    state_debug_columns: str = "timer,hp1,hp2,meter1,meter2",
    state_debug_norm_max: str = "99,160,160,104,96",
    **inputs,
):
    """StatePlay video flow loss plus clean-state Smooth-L1 loss."""
    if "lora" in inputs:
        pipe.clear_lora(verbose=0)
        pipe.load_lora(pipe.dit, state_dict=inputs["lora"], hotload=True, verbose=0)
    if "state" not in inputs:
        raise ValueError("[StatePlay state] inputs must include state; no zero fallback is used.")

    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    sigma = pipe.scheduler.sigmas[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    target_video = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    if "first_frame_latents" in inputs:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]

    f = inputs["input_latents"].shape[2]
    clean_state = downsample_state_to_latent(
        inputs["state"], f, sampling=state_sampling,
    ).to(device=pipe.device, dtype=pipe.torch_dtype)
    noise_state = torch.randn_like(clean_state)
    noisy_state = pipe.scheduler.add_noise(clean_state, noise_state, timestep)
    noisy_state[:, 0:1] = clean_state[:, 0:1]
    inputs["noisy_state"] = noisy_state

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    pred_video, pred_state = pipe.model_fn(**models, **inputs, timestep=timestep)

    if "first_frame_latents" in inputs:
        pred_video = pred_video[:, :, 1:]
        target_video = target_video[:, :, 1:]

    loss_video = torch.nn.functional.mse_loss(pred_video.float(), target_video.float())
    loss_video = loss_video * pipe.scheduler.training_weight(timestep)
    loss_state = torch.nn.functional.smooth_l1_loss(
        pred_state[:, 1:].float(), clean_state[:, 1:].float(),
    )
    loss_state = loss_state * pipe.scheduler.training_weight(timestep)
    if (
        state_debug_enabled
        and state_debug_path
        and state_debug_step is not None
        and int(state_debug_step) > 0
        and int(state_debug_step) % max(1, int(state_debug_interval)) == 0
        and _state_debug_rank0()
    ):
        with torch.no_grad():
            pred_clean_state = pred_state
            pred_clean_state[:, 0:1] = clean_state[:, 0:1]
            _append_state_debug_record(
                state_debug_path,
                int(state_debug_step),
                timestep,
                sigma,
                pred_clean_state,
                clean_state,
                state_debug_columns,
                state_debug_norm_max,
            )
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            f"[loss_components] video_loss={loss_video.detach().item():.6f} "
            f"state_loss={loss_state.detach().item():.6f} "
            f"video_w={float(video_loss_weight):.3f} state_w={float(state_loss_weight):.3f}",
            flush=True,
        )
    return float(video_loss_weight) * loss_video + float(state_loss_weight) * loss_state


class FullModelLogger(ModelLogger):  # noqa: F405
    """Saves the entire trainable dit state dict (full or scoped param training)."""
    def __init__(self, output_path):
        super().__init__(output_path, remove_prefix_in_ckpt="pipe.dit.")


class StatePlayTrainingModule(DiffusionTrainingModule):  # noqa: F405
    def __init__(
        self,
        profile,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        trainable_filter="",
        trainable_filter_exclude="",
        use_csv_prompt=True,
        prompt_column="prompt",
        resume_from_ckpt=None,
        action_dropout_prob=0.0,
        prompt_dropout_prob=0.0,
        state_loss_weight=1.0,
        video_loss_weight=1.0,
        state_sampling="end",
        verbose=False,
        state_debug_interval=1000,
        state_debug_path=None,
        state_debug_columns="timer,hp1,hp2,meter1,meter2",
        state_debug_norm_max="99,160,160,104,96",
    ):
        super().__init__()
        self.profile = profile
        self.state_loss_weight = state_loss_weight
        self.video_loss_weight = video_loss_weight
        self.state_sampling = state_sampling
        self.verbose = bool(verbose)
        self.state_debug_interval = state_debug_interval
        self.state_debug_path = state_debug_path
        self.state_debug_columns = state_debug_columns
        self.state_debug_norm_max = state_debug_norm_max
        self.state_debug_current_step = None
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing disabled — watch for OOM.")

        model_configs = self.parse_model_configs(
            model_paths, model_id_with_origin_paths,
            fp8_models=fp8_models, offload_models=offload_models, device=device,
        )
        tokenizer_config = (
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B",
                        origin_file_pattern="google/umt5-xxl/")
            if tokenizer_path is None else ModelConfig(tokenizer_path)
        )
        self.pipe = StatePlayTrainingPipeline.from_pretrained(
            torch_dtype=torch.bfloat16, device=device,
            model_configs=model_configs, tokenizer_config=tokenizer_config,
        )

        # Cold-start: build StatePlayModel and copy shape-matched weights from
        # the pretrained Wan2.2-TI2V-5B DiT. ActionModule keys stay at their
        # default (zero / xavier) initialization. Optional resume_from_ckpt
        # then overlays a previously-saved DiT state dict (no optimizer state).
        custom_dit = StatePlayModel(
            num_buttons=profile.num_buttons, state_dim=5, **WAN_MODEL_KWARGS,
        ).to(torch.bfloat16)
        self.pipe.model_fn = model_fn_stateplay
        custom_dit = self._transfer_weights(custom_dit, self.pipe.dit)
        if resume_from_ckpt:
            state = load_file(resume_from_ckpt)
            missing, unexpected = custom_dit.load_state_dict(state, strict=False)
            print(f"[Resume] loaded {len(state)} keys from {resume_from_ckpt} "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
            state_missing = [k for k in missing if k.startswith(("state_encoder.", "state_decoder."))]
            if state_missing:
                print("[Resume] state_encoder/state_decoder may be absent in a Wan backbone checkpoint.")
            has_state_branch = any(
                ".state_self_attn." in k or ".state_ffn." in k
                for k in state.keys()
            )
            if hasattr(custom_dit, "init_state_branch_from_pretrained") and not has_state_branch:
                copied = custom_dit.init_state_branch_from_pretrained(state)
                print(f"[Resume] StatePlay copied {copied} state-branch self_attn/ffn tensors from resume checkpoint.")
            del state
        self.pipe.dit = custom_dit.to(self.pipe.device)
        del custom_dit
        torch.cuda.empty_cache()

        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        if trainable_models:
            for name in trainable_models.split(","):
                module = self.pipe.get_module(self.pipe, name)
                if module is not None:
                    module.train()
                    module.requires_grad_(True)

        self._apply_trainable_filter(trainable_filter, trainable_filter_exclude)

        self.action_dropout_prob = action_dropout_prob
        self.prompt_dropout_prob = prompt_dropout_prob

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.use_csv_prompt = use_csv_prompt
        self.prompt_column = prompt_column

    @staticmethod
    def _transfer_weights(custom_model, pretrained_dit):
        pretrained = pretrained_dit.state_dict()
        custom = custom_model.state_dict()
        new = {k: v for k, v in pretrained.items()
               if k in custom and v.shape == custom[k].shape}
        missing, _ = custom_model.load_state_dict(new, strict=False)
        state_missing = [k for k in missing if k.startswith(("state_encoder.", "state_decoder."))]
        action_missing = [k for k in missing if k.startswith("action_embedders.")]
        print(f"[Cold-start] Transferred {len(new)}/{len(pretrained)} keys; "
              f"{len(missing)} keys remain at init "
              f"(action={len(action_missing)}, state={len(state_missing)}, other={len(missing) - len(action_missing) - len(state_missing)}).")
        if state_missing:
            print("[Cold-start] state_encoder/state_decoder keys are expected to remain at init.")
        if hasattr(custom_model, "init_state_branch_from_pretrained"):
            copied = custom_model.init_state_branch_from_pretrained(pretrained)
            print(f"[Cold-start] StatePlay copied {copied} state-branch self_attn/ffn tensors from Wan base weights.")
        return custom_model

    def _apply_trainable_filter(self, keep: str, drop: str):
        """Two complementary substring filters on dit.named_parameters().

        - keep: keep ONLY params whose name contains any of the comma-listed
                substrings (everything else frozen).
        - drop: freeze params whose name contains any of these substrings
                (everything else trainable).
        Both can be combined: keep is applied first, then drop further freezes.
        """
        if not (keep or drop):
            return
        keep_pats = [s.strip() for s in keep.split(",") if s.strip()]
        drop_pats = [s.strip() for s in drop.split(",") if s.strip()]
        for name, p in self.pipe.dit.named_parameters():
            if keep_pats and not any(pat in name for pat in keep_pats):
                p.requires_grad = False
                continue
            if drop_pats and any(pat in name for pat in drop_pats):
                p.requires_grad = False
        scalars = sum(p.numel() for p in self.pipe.dit.parameters() if p.requires_grad)
        tensors = sum(1 for p in self.pipe.dit.parameters() if p.requires_grad)
        print(f"[Trainable Filter] keep={keep_pats} drop={drop_pats} -> "
              f"{tensors} tensors trainable ({scalars:,} scalars)")

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for x in extra_inputs:
            if x == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif x == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif x in ("reference_image", "vace_reference_image"):
                inputs_shared[x] = data[x][0]
            else:
                inputs_shared[x] = data[x]
        return inputs_shared

    def get_pipeline_inputs(self, data):
        prompt = resolve_prompt(data, self.profile, self.use_csv_prompt, self.prompt_column)
        inputs_posi = {"prompt": prompt}
        inputs_nega = {}
        inputs_shared = {
            "input_video": data["video"],
            "keyboard_action": data["action"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        if "state" not in data:
                raise ValueError(
                    "[StatePlay state] dataset item does not contain `state`. "
                    "Use build_dataset(...)  with valid state columns."
                )
        inputs_shared["state"] = data["state"]
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)

        # CPU RNG #1: action_dropout. With prob p, zero out the entire keyboard
        # so the model learns a "fix_prompt + no-action" baseline instead of
        # memorizing per-clip action-sequence hashes. No-op when prob == 0.
        if self.action_dropout_prob > 0.0 and torch.rand(1).item() < self.action_dropout_prob:
            inputs_shared = inputs[0]
            ka = inputs_shared.get("keyboard_action", None)
            if ka is not None:
                inputs_shared["keyboard_action"] = torch.zeros_like(ka)

        # CPU RNG #2: prompt_dropout. With prob p, swap to unconditional prompt.
        # Inference --cfg_scale>1 then extrapolates the delta to amplify NPC.
        drop_prompt = (
            self.prompt_dropout_prob > 0.0
            and torch.rand(1).item() < self.prompt_dropout_prob
        )

        if drop_prompt:
            inputs_shared, inputs_posi, inputs_nega = inputs
            if "prompt" in inputs_posi:
                inputs_posi["prompt"] = ""
            inputs = (inputs_shared, inputs_posi, inputs_nega)

        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        inputs_shared, inputs_posi, inputs_nega = inputs
        return FlowMatchSFTLossWithStateWeighted(
            self.pipe,
            state_loss_weight=self.state_loss_weight,
            video_loss_weight=self.video_loss_weight,
            state_sampling=self.state_sampling,
            state_debug_enabled=self.verbose,
            state_debug_step=self.state_debug_current_step,
            state_debug_interval=self.state_debug_interval,
            state_debug_path=self.state_debug_path,
            state_debug_columns=self.state_debug_columns,
            state_debug_norm_max=self.state_debug_norm_max,
            **inputs_shared, **inputs_posi,
        )


def _bool_arg(s: str) -> bool:
    """argparse type for `--use_csv_prompt true|false` (case insensitive)."""
    return str(s).strip().lower() not in ("false", "0", "no", "off", "")


def build_parser():
    p = argparse.ArgumentParser(description="StatePlay training")
    p = add_general_config(p)  # noqa: F405
    p = add_video_size_config(p)  # noqa: F405
    p.add_argument("--tokenizer_path", type=str, default=None)
    p.add_argument("--max_timestep_boundary", type=float, default=1.0)
    p.add_argument("--min_timestep_boundary", type=float, default=0.0)
    p.add_argument("--initialize_model_on_cpu", default=False, action="store_true")
    p.add_argument("--action_hold_window", type=int, default=10,
                   help="Hold-last upsampling window for sparse parquet (10Hz->20fps). "
                        "Set 1 to disable.")
    # `--use_csv_prompt` accepts true/false explicitly; default is None so we
    # can fall back to the StatePlay profile default.
    p.add_argument("--use_csv_prompt", type=_bool_arg, default=None,
                   help="Read per-clip prompt from the CSV column. "
                        "Pass `--use_csv_prompt true|false` to override the StatePlay default.")
    p.add_argument("--prompt_column", type=str, default="prompt",
                   help="CSV column to read for the per-clip prompt.")
    p.add_argument("--trainable_filter", type=str, default="",
                   help="Comma-separated substrings; only DiT params whose name "
                        "contains ANY substring stay trainable. Empty = no filter.")
    p.add_argument("--trainable_filter_exclude", type=str, default="",
                   help="Comma-separated substrings; DiT params whose name "
                        "contains ANY substring are frozen. Empty = no filter.")
    p.add_argument("--resume_from_ckpt", type=str, default=None,
                   help="Path to a previously saved DiT safetensors (output of "
                        "FullModelLogger). Loaded into StatePlayModel after the "
                        "cold-start transfer_weights, before training starts. "
                        "Optimizer state is NOT resumed.")
    p.add_argument("--action_dropout_prob", type=float, default=0.0,
                   help="Per-step probability of zeroing keyboard_action (CFG-style).")
    p.add_argument("--prompt_dropout_prob", type=float, default=0.0,
                   help="Per-step probability of replacing the prompt with empty "
                        "string (CFG-style training). Default 0.0 (off).")
    p.add_argument("--state_columns", type=str, default="timer,hp1,hp2,meter1,meter2",
                   help="StatePlay state columns. Friendly aliases hp1/hp2/meter1/meter2 map to p1_hp/p2_hp/p1_meter/p2_meter when present.")
    p.add_argument("--state_norm_max", type=str, default="99,160,160,104,96",
                   help="Comma-separated maxima for normalizing StatePlay state to [-1, 1].")
    p.add_argument("--state_loss_weight", type=float, default=1.0,
                   help="Weight for StatePlay state flow-matching loss.")
    p.add_argument("--video_loss_weight", type=float, default=1.0,
                   help="Weight for StatePlay video flow-matching loss. Set 0 for state-only warmup.")
    p.add_argument("--state_sampling", type=str, default="end",
                   choices=("interpolation", "avg", "end"),
                   help="How StatePlay downsamples raw per-frame state to latent time: "
                        "linear interpolation, adaptive average pooling, or "
                        "the last sample in each adaptive bin.")
    p.add_argument("--verbose", type=_bool_arg, default=False,
                   help="If true, append state prediction-vs-target tables every state_debug_interval steps.")
    p.add_argument("--state_debug_interval", type=int, default=1000,
                   help="Step interval for --verbose state prediction logging.")
    p.add_argument("--state_debug_path", type=str, default=None,
                   help="Output text file for --verbose state prediction logging.")
    p.add_argument("--max_train_steps", type=int, default=None,
                   help="Stop after this many optimizer steps; useful for smoke tests.")
    p.add_argument("--reserve_cuda_memory_gb", type=float, default=0.0,
                   help="Temporarily allocate this many GiB on each rank, then release the tensor while keeping PyTorch CUDA cache reserved for training.")
    p.add_argument("--save_full_state", default=False, action="store_true",
                   help="At each save_steps boundary, also dump "
                        "accelerator.save_state() to <output>/state-<step>/ "
                        "alongside the existing step-<step>.safetensors. "
                        "Enables seamless --resume_state.")
    p.add_argument("--resume_state", type=str, default=None,
                   help="Path to a state-<N>/ directory written by "
                        "--save_full_state. Loaded via accelerator.load_state() "
                        "AFTER prepare() — restores optimizer moments, LR "
                        "scheduler, RNG, and DataLoader state. Mutually "
                        "exclusive with --resume_from_ckpt.")
    return p


def build_dataset(args, profile):
    # StatePlay dataset path. Only the action parquet operator is special; the rest
    # (video + first-frame image) come from UnifiedDataset's default video
    # operator. StatePlay does not consume audio or animate streams.
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height, width=args.width,
            height_division_factor=16, width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4, time_division_remainder=1,
        ),
        special_operator_map={
            "action": get_action_op(profile, args.dataset_base_path, args.num_frames,
                                    hold_window=args.action_hold_window),
        },
    )
    dataset = StateAugmentedDataset(
        dataset,
        metadata_path=args.dataset_metadata_path,
        base_path=args.dataset_base_path,
        num_frames=args.num_frames,
        state_columns=args.state_columns,
        state_norm_max=args.state_norm_max,
    )
    return dataset


if __name__ == "__main__":
    args = build_parser().parse_args()
    profile = SF3
    # Resolve StatePlay defaults.
    if args.use_csv_prompt is None:
        args.use_csv_prompt = profile.default_use_csv_prompt
    if args.resume_state and args.resume_from_ckpt:
        raise SystemExit(
            "--resume_state and --resume_from_ckpt are mutually exclusive: "
            "the former restores optimizer/LR/RNG/dataloader (seamless); the "
            "latter overlays a bare DiT safetensors (optimizer not restored)."
        )
    print(f"[StatePlay] profile={profile.name} ({profile.description})")
    print(f"[StatePlay] use_csv_prompt={args.use_csv_prompt} prompt_column={args.prompt_column}")
    print(f"[StatePlay] num_buttons={profile.num_buttons} button_cols={list(profile.button_cols)}")
    print(f"[StatePlay state] state_columns={args.state_columns} state_norm_max={args.state_norm_max} state_loss_weight={args.state_loss_weight} video_loss_weight={args.video_loss_weight} state_sampling={args.state_sampling} loss=clean_smooth_l1")

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(
            find_unused_parameters=args.find_unused_parameters)],
    )
    _reserve_cuda_memory(args.reserve_cuda_memory_gb)
    dataset = build_dataset(args, profile)
    model = StatePlayTrainingModule(
        profile=profile,
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        trainable_filter=args.trainable_filter,
        trainable_filter_exclude=args.trainable_filter_exclude,
        use_csv_prompt=args.use_csv_prompt,
        prompt_column=args.prompt_column,
        resume_from_ckpt=args.resume_from_ckpt,
        action_dropout_prob=args.action_dropout_prob,
        prompt_dropout_prob=args.prompt_dropout_prob,
        state_loss_weight=args.state_loss_weight,
        video_loss_weight=args.video_loss_weight,
        state_sampling=args.state_sampling,
        verbose=args.verbose,
        state_debug_interval=args.state_debug_interval,
        state_debug_path=args.state_debug_path,
        state_debug_columns=args.state_columns,
        state_debug_norm_max=args.state_norm_max,
    )
    model_logger = FullModelLogger(args.output_path)
    # Training paths use the local launcher with seamless-resume hooks.
    # Data-process paths keep the upstream launcher unchanged.
    launcher_map = {
        "sft:data_process": launch_data_process_task,  # noqa: F405
        "direct_distill:data_process": launch_data_process_task,  # noqa: F405
        "sft": launch_training_task_with_state,
        "sft:train": launch_training_task_with_state,
        "direct_distill": launch_training_task_with_state,
        "direct_distill:train": launch_training_task_with_state,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)

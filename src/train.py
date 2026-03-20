import json
import os
import time
import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from accelerate import Accelerator
from diffusers import DDPMScheduler, StableDiffusionXLPipeline, AutoencoderKL
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model, PeftModel

from config import TrainConfig, get_processed_root
from dataset import Text2ImageDataset, load_sampling_weights
from mlflow_utils import MLflowTracker, ExperimentTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_random_seed() -> int:
    return int.from_bytes(os.urandom(4), "big") % (2**31)


def count_parameters(model, only_trainable: bool = True) -> int:
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


KAGGLE_DATASET_ROOT = Path("/kaggle/input/datasets/souissiyoussef/diffusion-text2image")
KAGGLE_RESULTS_ROOT = KAGGLE_DATASET_ROOT / "output" / "results"
KAGGLE_CHECKPOINTS_DIR = KAGGLE_RESULTS_ROOT / "checkpoints"
KAGGLE_LATEST_CHECKPOINT = KAGGLE_CHECKPOINTS_DIR / "checkpoint-15000"
EXTRA_CHECKPOINT_SEARCH_PATHS = [KAGGLE_CHECKPOINTS_DIR]


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    search_dirs = [checkpoint_dir] + EXTRA_CHECKPOINT_SEARCH_PATHS
    all_checkpoints = []
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        found = [d for d in search_dir.glob("checkpoint-*") if d.is_dir()]
        all_checkpoints.extend(found)

    if not all_checkpoints:
        return None

    all_checkpoints = sorted(
        all_checkpoints,
        key=lambda x: int(x.name.split("-")[1]),
        reverse=True,
    )
    best = all_checkpoints[0]
    logger.info(f"Checkpoint found: {best}")
    return best


def load_checkpoint_step(checkpoint_path: Path) -> int:
    meta_file = checkpoint_path / "checkpoint_meta.json"
    if meta_file.exists():
        with open(meta_file, "r") as f:
            meta = json.load(f)
        return int(meta.get("global_step", 0))
    try:
        return int(checkpoint_path.name.split("-")[1])
    except (IndexError, ValueError):
        return 0


def prepare_models_and_lora(config, device: str, resume_from_checkpoint: Optional[Path] = None):
    logger.info(f"Loading base model: {config.model_id}")

    vae = AutoencoderKL.from_pretrained(config.vae_model_id, torch_dtype=torch.float16)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        config.model_id,
        vae=vae,
        torch_dtype=torch.float16 if config.fp16 else torch.float32,
        variant="fp16" if config.fp16 else None,
        use_safetensors=True,
    )

    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)

    unet = pipe.unet

    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        init_lora_weights=config.lora.init_lora_weights,
        use_dora=config.lora.use_dora,
        use_rslora=config.lora.use_rslora,
    )

    if resume_from_checkpoint:
        print(f"Resuming LoRA from checkpoint: {resume_from_checkpoint}", flush=True)
        try:
            unet = PeftModel.from_pretrained(unet, str(resume_from_checkpoint), is_trainable=True)
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}. Applying LoRA from scratch.")
            unet = get_peft_model(unet, lora_config)
    else:
        unet = get_peft_model(unet, lora_config)

    total_params = count_parameters(unet, only_trainable=False)
    trainable_params = count_parameters(unet, only_trainable=True)
    print(f"UNet total params:     {total_params:,}", flush=True)
    print(f"LoRA trainable params: {trainable_params:,}", flush=True)

    if trainable_params > 20_000_000:
        raise RuntimeError(f"LoRA too large ({trainable_params:,} > 20M). Reduce lora_r.")

    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if config.use_xformers:
        try:
            if hasattr(unet, "enable_xformers_memory_efficient_attention"):
                unet.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(f"xFormers not available: {e}")

    if config.enable_attention_slicing:
        pipe.enable_attention_slicing()

    pipe.unet = unet
    return pipe


def encode_prompts_sdxl(pipe, prompts: list, device: str):
    text_inputs_1 = pipe.tokenizer(
        prompts,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    text_inputs_2 = pipe.tokenizer_2(
        prompts,
        padding="max_length",
        max_length=pipe.tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        out_1 = pipe.text_encoder(text_inputs_1.input_ids, output_hidden_states=True)
        embeds_1 = out_1.hidden_states[-2]          # (B, 77, 768)

        out_2 = pipe.text_encoder_2(text_inputs_2.input_ids, output_hidden_states=True)
        embeds_2 = out_2.hidden_states[-2]          # (B, 77, 1280)
        pooled_embeds = out_2[0]                    # (B, 1280)

    prompt_embeds = torch.cat([embeds_1, embeds_2], dim=-1)  # (B, 77, 2048)
    return prompt_embeds, pooled_embeds


class TrainingState:

    def __init__(self, config):
        self.config = config
        self.global_step = 0
        self.best_loss = float("inf")
        self.last_checkpoint_step = 0

    def resume_from_step(self, step: int):
        self.global_step = step
        self.last_checkpoint_step = step
        remaining = self.config.max_train_steps - step
        print(f"[Resume] global_step={step} — remaining steps: {remaining}", flush=True)

    def should_save_checkpoint(self) -> bool:
        return (self.global_step - self.last_checkpoint_step) >= self.config.save_steps

    def save_checkpoint(self, pipe, accelerator, output_dir: Path):
        print(f"[Checkpoint] Saving step {self.global_step}...", flush=True)
        checkpoint_dir = output_dir / f"checkpoint-{self.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            unwrapped_unet = accelerator.unwrap_model(pipe.unet)
            unwrapped_unet.save_pretrained(checkpoint_dir)
            with open(checkpoint_dir / "checkpoint_meta.json", "w") as f:
                json.dump({"global_step": self.global_step, "best_loss": self.best_loss}, f, indent=2)
            print(f"[Checkpoint] Saved to {checkpoint_dir}", flush=True)

        self.last_checkpoint_step = self.global_step
        self._cleanup_old_checkpoints(output_dir)

    def _cleanup_old_checkpoints(self, output_dir: Path):
        keep_last = self.config.max_checkpoints_total
        checkpoints = sorted(
            [d for d in output_dir.glob("checkpoint-*")],
            key=lambda x: int(x.name.split("-")[1]),
            reverse=True,
        )
        for old_ckpt in checkpoints[keep_last:]:
            import shutil
            shutil.rmtree(old_ckpt)
            logger.info(f"Deleted old checkpoint: {old_ckpt.name}")


def run_training(config=None):
    if config is None:
        config = TrainConfig

    config.resolve_paths()
    set_seed(config.seed)

    checkpoint_dir = Path(config.output_dir)
    resume_checkpoint = find_latest_checkpoint(checkpoint_dir)

    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}", flush=True)
    else:
        print("No checkpoint found. Starting from scratch.", flush=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision="fp16" if config.fp16 else "no",
        log_with=None,
        project_dir=config.logging_dir,
    )
    device = accelerator.device
    print(f"Training on: {device}", flush=True)

    mlflow_tracker = MLflowTracker(
        Path(config.mlflow_dir),
        config.mlflow_experiment_name,
        config.mlflow_run_name,
    )
    mlflow_tracker.start_run(tags={
        "model_id": config.model_id,
        "phase": "lora_sdxl",
        "framework": "pytorch+accelerate",
        "resumed": str(resume_checkpoint is not None),
    })
    mlflow_tracker.log_params(config.to_dict())

    exp_tracker = ExperimentTracker(Path(config.output_dir))

    print("Loading dataset...", flush=True)
    train_dataset = Text2ImageDataset("train", data_root=get_processed_root(), validate=False)
    print(f"Training samples: {len(train_dataset)}", flush=True)

    sampler = None
    shuffle = True
    if config.use_weighted_sampling:
        weights = load_sampling_weights(get_processed_root())
        if weights and weights[0]:
            class_weights_dict = weights[0]
            sample_weights = [
                class_weights_dict.get(s["class_name"], 1.0)
                for s in train_dataset.samples
            ]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
            shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )
    print(f"DataLoader: {len(train_loader)} batches", flush=True)

    pipe = prepare_models_and_lora(config, device, resume_from_checkpoint=resume_checkpoint)
    pipe.to(device)

    noise_scheduler = DDPMScheduler.from_pretrained(config.model_id, subfolder="scheduler")

    trainable_params_list = [p for p in pipe.unet.parameters() if p.requires_grad]
    print(f"Trainable params for optimizer: {sum(p.numel() for p in trainable_params_list):,}", flush=True)

    if len(trainable_params_list) == 0:
        raise RuntimeError("No trainable parameters found in UNet. Check LoRA setup.")

    if getattr(config, "use_8bit_adam", False):
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                trainable_params_list,
                lr=config.learning_rate,
                weight_decay=config.adam_weight_decay,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_epsilon,
            )
            print("Using 8-bit AdamW (bitsandbytes)", flush=True)
        except ImportError:
            print("bitsandbytes not found — falling back to AdamW", flush=True)
            optimizer = torch.optim.AdamW(
                trainable_params_list,
                lr=config.learning_rate,
                weight_decay=config.adam_weight_decay,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_epsilon,
            )
    else:
        optimizer = torch.optim.AdamW(
            trainable_params_list,
            lr=config.learning_rate,
            weight_decay=config.adam_weight_decay,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
        )
        print("Using standard AdamW", flush=True)

    lr_scheduler = get_scheduler(
        config.scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=config.max_train_steps,
    )

    pipe.unet, train_loader, optimizer, lr_scheduler = accelerator.prepare(
        pipe.unet, train_loader, optimizer, lr_scheduler,
    )

    state = TrainingState(config)

    resumed_step = 0
    if resume_checkpoint:
        resumed_step = load_checkpoint_step(resume_checkpoint)
        if resumed_step > 0:
            state.resume_from_step(resumed_step)

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.logging_dir, exist_ok=True)
    pipe.unet.train()

    training_start_time = time.time()
    nan_count = 0
    logged_steps = set()

    unet_dtype = torch.float16 if config.fp16 else torch.float32
    add_time_ids = torch.tensor(
        [[1024, 1024, 0, 0, 1024, 1024]],
        dtype=unet_dtype,
        device=device,
    )

    print("=" * 70, flush=True)
    print(f"Starting training — max_steps={config.max_train_steps}", flush=True)
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps * accelerator.num_processes}", flush=True)
    print("=" * 70, flush=True)

    for epoch in range(1000):
        print(f"\nEpoch {epoch + 1}", flush=True)

        for batch_idx, batch in enumerate(train_loader):

            with accelerator.accumulate(pipe.unet):

                pixel_values = batch["pixel_values"].to(device, dtype=pipe.vae.dtype)

                with torch.no_grad():
                    latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * pipe.vae.config.scaling_factor

                noise = torch.randn_like(latents)

                if getattr(config, "noise_offset", 0.0) > 0:
                    noise = noise + config.noise_offset * torch.randn(
                        latents.shape[0], latents.shape[1], 1, 1,
                        device=device, dtype=noise.dtype,
                    )

                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=device,
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                prompts = batch["prompt"]
                prompt_embeds, pooled_prompt_embeds = encode_prompts_sdxl(pipe, prompts, device)
                prompt_embeds = prompt_embeds.to(unet_dtype)
                pooled_prompt_embeds = pooled_prompt_embeds.to(unet_dtype)

                batch_size = latents.shape[0]
                time_ids_batch = add_time_ids.repeat(batch_size, 1)

                noise_pred = pipe.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={
                        "text_embeds": pooled_prompt_embeds,
                        "time_ids": time_ids_batch,
                    },
                ).sample

                loss = torch.nn.functional.mse_loss(
                    noise_pred.float(), noise.float(), reduction="mean"
                )

                if torch.isnan(loss):
                    nan_count += 1
                    logger.warning(f"NaN detected at batch {batch_idx} (total={nan_count}). Skipping.")
                    optimizer.zero_grad()
                    continue

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(pipe.unet.parameters(), config.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                state.global_step += 1

            if (state.global_step > 0
                    and state.global_step % config.logging_steps == 0
                    and state.global_step not in logged_steps):
                logged_steps.add(state.global_step)
                loss_scalar = accelerator.gather(loss.detach()).mean().item()
                current_lr = optimizer.param_groups[0]["lr"]

                exp_tracker.log_training_step(state.global_step, loss_scalar, current_lr)

                elapsed = time.time() - training_start_time
                steps_this_session = state.global_step - (resumed_step if resume_checkpoint and resumed_step > 0 else 0)
                steps_per_sec = max(steps_this_session, 1) / max(elapsed, 1.0)
                eta_hours = (config.max_train_steps - state.global_step) / steps_per_sec / 3600

                msg = (
                    f"[Step {state.global_step:5d}/{config.max_train_steps}] "
                    f"Loss: {loss_scalar:.6f} | LR: {current_lr:.2e} | ETA: {eta_hours:.1f}h"
                )
                logger.info(msg)
                print(msg, flush=True)

                if accelerator.is_main_process:
                    mlflow_tracker.log_metric("training/loss", loss_scalar, step=state.global_step)
                    mlflow_tracker.log_metric("training/learning_rate", current_lr, step=state.global_step)

                if loss_scalar < state.best_loss:
                    state.best_loss = loss_scalar

            if state.should_save_checkpoint() and accelerator.is_main_process:
                state.save_checkpoint(pipe, accelerator, Path(config.output_dir))

            if state.global_step >= config.max_train_steps:
                print(f"Reached max_train_steps ({config.max_train_steps}).", flush=True)
                break

        if state.global_step >= config.max_train_steps:
            break

    if accelerator.is_main_process:
        final_dir = Path(config.output_dir) / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        unwrapped_unet = accelerator.unwrap_model(pipe.unet)
        unwrapped_unet.save_pretrained(final_dir)
        config.save_to_json(final_dir / "config.json")
        exp_tracker.save_history(Path(config.output_dir) / "training_history.json")
        exp_tracker.save_best_metrics(Path(config.output_dir) / "best_metrics.json")

        total_hours = (time.time() - training_start_time) / 3600
        print("\n" + "=" * 70, flush=True)
        print("TRAINING COMPLETE!", flush=True)
        print(f"Steps:       {state.global_step}/{config.max_train_steps}", flush=True)
        print(f"Best loss:   {state.best_loss:.6f}", flush=True)
        print(f"Total time:  {total_hours:.2f}h", flush=True)
        print(f"NaN batches: {nan_count}", flush=True)
        print(f"Final ckpt:  {final_dir}", flush=True)
        print("=" * 70 + "\n", flush=True)

        pct = (state.global_step / config.max_train_steps) * 100
        if state.global_step >= config.max_train_steps * 0.95:
            print("Training COMPLETE.", flush=True)
        elif state.global_step >= config.max_train_steps * 0.80:
            print("WARNING: Incomplete but >80% done (resumable).", flush=True)
        else:
            print(f"ERROR: Only {pct:.1f}% complete!", flush=True)

    if accelerator.is_main_process:
        mlflow_tracker.log_dict(exp_tracker.get_summary(), filename="training_summary.json")
        mlflow_tracker.end_run()

    return pipe, exp_tracker


if __name__ == "__main__":
    pipe, tracker = run_training()
    logger.info("SDXL LoRA training pipeline completed successfully.")

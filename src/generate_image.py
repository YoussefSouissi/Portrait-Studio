"""
Inference utilities for SDXL + LoRA: single image, batch, and qualitative grid generation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

from config import TrainConfig
from evaluate import load_lora_from_checkpoint
from inference_config import InferenceConfig
from improved_prompts import IMPROVED_PROMPTS


def get_random_seed() -> int:
    return int.from_bytes(os.urandom(4), "big") % (2**31)


def load_model(checkpoint_path: str | Path, model_id: str | None = None) -> StableDiffusionXLPipeline:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    base_model_id = model_id or TrainConfig.model_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    vae = AutoencoderKL.from_pretrained(
        TrainConfig.vae_model_id,
        torch_dtype=dtype,
    )

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_id,
        vae=vae,
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
        use_safetensors=True,
    )
    pipe.to(device)

    pipe = load_lora_from_checkpoint(pipe, str(checkpoint_path))

    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.set_progress_bar_config(disable=True)

    return pipe


def generate_image(
    pipeline: StableDiffusionXLPipeline,
    prompt: str,
    config: InferenceConfig,
) -> Tuple[Image.Image, int]:
    device = pipeline.device

    seed_used = get_random_seed() if (config.use_random_seed or config.seed is None) else int(config.seed)
    generator = torch.Generator(device=device).manual_seed(seed_used)

    with torch.no_grad():
        out = pipeline(
            prompt,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            generator=generator,
            height=config.height,
            width=config.width,
        )

    if not out.images:
        raise RuntimeError("Pipeline returned no images.")

    return out.images[0], seed_used


def generate_multiple(
    pipeline: StableDiffusionXLPipeline,
    prompts: List[str],
    config: InferenceConfig,
    num_per_prompt: int = 3,
) -> List[Tuple[Image.Image, int, str]]:
    results: List[Tuple[Image.Image, int, str]] = []

    cfg = InferenceConfig(
        guidance_scale=config.guidance_scale,
        num_inference_steps=config.num_inference_steps,
        use_random_seed=config.use_random_seed,
        seed=config.seed,
        height=config.height,
        width=config.width,
    )

    for prompt in prompts:
        for _ in range(num_per_prompt):
            if cfg.use_random_seed:
                cfg.seed = get_random_seed()
            img, seed_used = generate_image(pipeline, prompt, cfg)
            results.append((img, seed_used, prompt))

    return results


def _make_image_grid(images: List[Image.Image], rows: int, cols: int) -> Image.Image:
    if not images:
        raise ValueError("No images provided.")
    if rows * cols != len(images):
        raise ValueError(f"rows×cols ({rows}×{cols}) != len(images) ({len(images)}).")
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h))
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        grid.paste(img, (c * w, r * h))
    return grid


def generate_qualitative_grid(
    pipeline: StableDiffusionXLPipeline,
    config: InferenceConfig,
    num_prompts: int = 4,
    num_per_prompt: int = 2,
    save_path: str | Path | None = None,
) -> Tuple[Image.Image, List[Tuple[str, int]]]:
    if num_prompts > len(IMPROVED_PROMPTS):
        raise ValueError(f"num_prompts={num_prompts} > len(IMPROVED_PROMPTS)={len(IMPROVED_PROMPTS)}")

    selected_prompts = IMPROVED_PROMPTS[:num_prompts]
    results = generate_multiple(pipeline, selected_prompts, config, num_per_prompt=num_per_prompt)

    images = [img for (img, _seed, _p) in results]
    meta: List[Tuple[str, int]] = [(p, s) for (_img, s, p) in results]

    grid = _make_image_grid(images, rows=num_prompts, cols=num_per_prompt)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        grid.save(save_path, format="PNG")

    return grid, meta

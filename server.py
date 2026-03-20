import os
import gc
import json
import hashlib
import time
import tempfile
import logging
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from flask import Flask, request, send_file, jsonify, send_from_directory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SDXL_ID   = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_ID    = "madebyollin/sdxl-vae-fp16-fix"
LORA_ID   = "YoussefSouissi/celeba-lora"
MODELS_DIR = Path(os.environ.get("HF_HOME", "./models/hf_cache"))
DIST_DIR   = Path(__file__).parent / "front-end" / "dist"

NEGATIVE = (
    "deformed face, mutation, extra nose, double face, two faces, "
    "blurry, low quality, ugly, disfigured, bad anatomy, "
    "extra limbs, missing nose, cropped face, out of frame, "
    "watermark, text, jpeg artifacts, oversaturated"
)

# Global state
pipe = None
hw   = None


# ── Hardware detection ────────────────────────────────────────────────────────

def detect_hardware() -> dict:
    if not torch.cuda.is_available():
        log.warning("No GPU detected — running on CPU.")
        return {
            "device": "cpu",
            "dtype": torch.float32,
            "resolution": 512,
            "profile": "cpu",
            "vram_gb": 0.0,
            "gpu_name": "None (CPU only)",
        }

    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    gpu_name = torch.cuda.get_device_name(0)

    if vram_gb >= 8:
        profile, resolution = "high", 1024
    elif vram_gb >= 5:
        profile, resolution = "mid", 1024
    elif vram_gb >= 3:
        profile, resolution = "low", 768
    else:
        profile, resolution = "very_low", 512

    return {
        "device": "cuda",
        "dtype": torch.float16,
        "resolution": resolution,
        "profile": profile,
        "vram_gb": round(vram_gb, 1),
        "gpu_name": gpu_name,
    }


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(hw: dict):
    from diffusers import StableDiffusionXLPipeline, AutoencoderKL, DPMSolverMultistepScheduler
    from peft import PeftModel
    from huggingface_hub import hf_hub_download

    os.environ["HF_HOME"] = str(MODELS_DIR)

    log.info("=" * 60)
    log.info("GPU      : %s", hw["gpu_name"])
    log.info("VRAM     : %s GB", hw["vram_gb"])
    log.info("Profile  : %s", hw["profile"])
    log.info("Resolut. : %dx%d", hw["resolution"], hw["resolution"])
    log.info("=" * 60)

    log.info("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=hw["dtype"])

    log.info("Loading SDXL base model...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_ID,
        vae=vae,
        torch_dtype=hw["dtype"],
        variant="fp16" if hw["device"] == "cuda" else None,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )

    log.info("Loading LoRA weights...")
    config_path  = hf_hub_download(repo_id=LORA_ID, filename="adapter_config.json")
    weights_path = hf_hub_download(repo_id=LORA_ID, filename="adapter_model.safetensors")

    with open(config_path) as f:
        cfg = json.load(f)
    for field in ["corda_config", "eva_config", "qalora_group_size", "lora_bias",
                  "target_parameters", "trainable_token_indices", "use_qalora", "exclude_modules"]:
        cfg.pop(field, None)

    tmp = tempfile.mkdtemp()
    with open(os.path.join(tmp, "adapter_config.json"), "w") as f:
        json.dump(cfg, f)
    os.symlink(weights_path, os.path.join(tmp, "adapter_model.safetensors"))

    pipe.unet = PeftModel.from_pretrained(pipe.unet, tmp, is_trainable=False)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )

    # Apply memory optimizations based on hardware profile
    profile = hw["profile"]
    if profile in ("cpu", "very_low"):
        pipe.enable_sequential_cpu_offload()
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
    elif profile == "low":
        pipe.enable_sequential_cpu_offload()
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
    elif profile == "mid":
        pipe.to(hw["device"])
        pipe.enable_attention_slicing(1)
        pipe.enable_vae_slicing()
    else:  # high
        pipe.to(hw["device"])
        pipe.enable_attention_slicing(1)

    gc.collect()
    if hw["device"] == "cuda":
        torch.cuda.empty_cache()

    log.info("Model ready.")
    return pipe


# ── Inference ─────────────────────────────────────────────────────────────────

def make_seed(prompt: str) -> int:
    raw = f"{prompt}{time.time_ns()}{os.urandom(8).hex()}"
    return int(hashlib.sha256(raw.encode()).hexdigest(), 16) % (2 ** 31)


def generate_image(prompt: str) -> tuple[Image.Image, int]:
    seed = make_seed(prompt)
    with torch.no_grad():
        result = pipe(
            prompt.strip(),
            negative_prompt=NEGATIVE,
            generator=torch.Generator(device=hw["device"]).manual_seed(seed),
            num_inference_steps=50,
            guidance_scale=9.0,
            height=hw["resolution"],
            width=hw["resolution"],
        )
    gc.collect()
    if hw["device"] == "cuda":
        torch.cuda.empty_cache()
    return result.images[0], seed


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route("/api/status")
def api_status():
    return jsonify({
        "profile":      hw["profile"] if hw else "unknown",
        "gpu_name":     hw["gpu_name"] if hw else "unknown",
        "vram_gb":      hw["vram_gb"] if hw else 0,
        "resolution":   hw["resolution"] if hw else 1024,
        "model_loaded": pipe is not None,
    })


@app.route("/api/generate", methods=["POST"])
def api_generate():
    if pipe is None:
        return "Model not loaded yet", 503

    data   = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return "No prompt provided", 400

    try:
        img, seed = generate_image(prompt)
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        response = send_file(buf, mimetype="image/png")
        response.headers["X-Seed"] = str(seed)
        response.headers["Access-Control-Expose-Headers"] = "X-Seed"
        return response
    except Exception as e:
        log.error("Generation error: %s", e)
        return str(e), 500


# Serve React SPA
@app.route("/")
def serve_index():
    return send_from_directory(str(DIST_DIR), "index.html")


@app.route("/<path:path>")
def serve_static(path):
    target = DIST_DIR / path
    if target.exists() and target.is_file():
        return send_from_directory(str(DIST_DIR), path)
    return send_from_directory(str(DIST_DIR), "index.html")


# ── Entry point ───────────────────────────────────────────────────────────────

def start(port: int = 8000):
    global pipe, hw

    hw   = detect_hardware()
    pipe = load_model(hw)

    log.info("Starting server on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    start()

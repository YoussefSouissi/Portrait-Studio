"""
Downloads all required models into the local cache.
Run once before starting the app, or let app.py call it automatically.
"""

import os
import sys
from pathlib import Path

MODELS_DIR = Path(os.environ.get("HF_HOME", "./models/hf_cache"))
SDXL_ID   = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_ID    = "madebyollin/sdxl-vae-fp16-fix"
LORA_ID   = "YoussefSouissi/celeba-lora"


def login_huggingface():
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("  No HF_TOKEN provided — downloading anonymously (may be slower).")
        return
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        print("  HuggingFace login successful.")
    except Exception as e:
        print(f"  HuggingFace login failed (continuing): {e}")


def is_cached(repo_id: str) -> bool:
    """Check if a repo is already in the HuggingFace cache."""
    from huggingface_hub import scan_cache_dir
    try:
        cache = scan_cache_dir(MODELS_DIR)
        return any(r.repo_id == repo_id for r in cache.repos)
    except Exception:
        return False


def download_all():
    from huggingface_hub import snapshot_download, hf_hub_download

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(MODELS_DIR)

    print("\n" + "=" * 60)
    print("  Portrait Studio — Model Download")
    print("=" * 60)

    login_huggingface()

    # SDXL base
    if is_cached(SDXL_ID):
        print(f"\n[1/3] SDXL base model — already cached, skipping.")
    else:
        print(f"\n[1/3] Downloading SDXL base model (~6.5 GB)...")
        snapshot_download(
            SDXL_ID,
            cache_dir=str(MODELS_DIR),
            ignore_patterns=["*.bin", "*.ot", "flax_model*", "tf_model*"],
        )
        print("      Done.")

    # Fixed VAE
    if is_cached(VAE_ID):
        print(f"\n[2/3] Fixed VAE — already cached, skipping.")
    else:
        print(f"\n[2/3] Downloading fixed VAE (~335 MB)...")
        snapshot_download(VAE_ID, cache_dir=str(MODELS_DIR))
        print("      Done.")

    # LoRA weights
    if is_cached(LORA_ID):
        print(f"\n[3/3] LoRA weights — already cached, skipping.")
    else:
        print(f"\n[3/3] Downloading LoRA weights (~50 MB)...")
        hf_hub_download(LORA_ID, "adapter_config.json",       cache_dir=str(MODELS_DIR))
        hf_hub_download(LORA_ID, "adapter_model.safetensors", cache_dir=str(MODELS_DIR))
        print("      Done.")

    print("\n" + "=" * 60)
    print("  All models ready.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    download_all()

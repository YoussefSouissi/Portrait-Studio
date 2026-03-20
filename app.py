"""
Portrait Studio — Local entry point.
Handles model download (if needed) then starts the Flask + React server.
"""

import os
import sys
from pathlib import Path

PORT      = int(os.environ.get("PORT", 8000))
MODELS_DIR = Path(os.environ.get("HF_HOME", "./models/hf_cache"))


def ensure_models():
    """Download models if not already cached."""
    from download_model import download_all, is_cached, SDXL_ID, VAE_ID, LORA_ID
    if not (is_cached(SDXL_ID) and is_cached(VAE_ID) and is_cached(LORA_ID)):
        print("Some models are missing. Starting download...")
        download_all()
    else:
        print("All models already cached.")


def main():
    print("\n" + "=" * 60)
    print("  Portrait Studio")
    print("=" * 60)

    # Step 1: ensure models are downloaded
    ensure_models()

    # Step 2: start the server (loads model + starts Flask)
    print(f"\nLoading model into memory... (this may take a few minutes)")
    print(f"\nOnce ready, open:  http://localhost:{PORT}\n")

    from server import start
    start(port=PORT)


if __name__ == "__main__":
    main()

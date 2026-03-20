from pathlib import Path
import json
import logging
from typing import Dict, List, Optional
import torch
import numpy as np
from PIL import Image
from collections import defaultdict

try:
    from config import get_results_root, EvalConfig
except Exception:
    EvalConfig = None

logger = logging.getLogger(__name__)


def load_lora_from_checkpoint(pipe, checkpoint_path: str):
    from peft import PeftModel
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"Loading LoRA from: {checkpoint_path}")
    pipe.unet = PeftModel.from_pretrained(pipe.unet, str(checkpoint_path), adapter_name="default")
    print("LoRA loaded successfully.")
    return pipe


_inception_cache = {}


def get_inception_model(device: str):
    if device not in _inception_cache:
        from torchvision.models import inception_v3
        model = inception_v3(weights="IMAGENET1K_V1", transform_input=False, aux_logits=True)
        model.fc = torch.nn.Identity()
        model.eval().to(device)
        _inception_cache[device] = model
    return _inception_cache[device]


def extract_features(images, device: str = "cuda", batch_size: int = 32) -> Optional[np.ndarray]:
    from torchvision import transforms
    model = get_inception_model(device)
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    feats = []
    if isinstance(images, (list, tuple)):
        for i in range(0, len(images), batch_size):
            batch_imgs = []
            for img_path in images[i:i + batch_size]:
                try:
                    img = Image.open(img_path).convert("RGB")
                    batch_imgs.append(preprocess(img))
                except Exception as e:
                    logger.warning(f"Skip {img_path}: {e}")
            if batch_imgs:
                x = torch.stack(batch_imgs).to(device)
                with torch.no_grad():
                    out = model(x)
                    if out.dim() > 2:
                        out = out.view(out.shape[0], -1)
                feats.append(out.cpu().numpy())

    return np.concatenate(feats, axis=0) if feats else None


# Aliases for notebook compatibility
extract_features_with_inception = extract_features


def sqrtm_approx(cov: np.ndarray) -> np.ndarray:
    try:
        from scipy import linalg
        return linalg.sqrtm(cov.astype(np.float64)).real
    except Exception:
        w, v = np.linalg.eigh(cov + 1e-6 * np.eye(cov.shape[0]))
        w = np.maximum(w, 0)
        return (v * np.sqrt(w)) @ v.T


def compute_fid(real_features: np.ndarray, fake_features: np.ndarray, eps: float = 1e-6) -> float:
    if real_features.size == 0 or fake_features.size == 0:
        return float("nan")

    mu_r = real_features.mean(axis=0)
    mu_f = fake_features.mean(axis=0)
    sigma_r = np.cov(real_features, rowvar=False) + eps * np.eye(real_features.shape[1])
    sigma_f = np.cov(fake_features, rowvar=False) + eps * np.eye(fake_features.shape[1])

    diff = mu_r - mu_f
    covmean = sqrtm_approx(sigma_r @ sigma_f)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma_r) + np.trace(sigma_f) - 2 * np.trace(covmean))
    return max(0.0, fid)


# Alias for notebook compatibility
compute_fid_from_features = compute_fid


def compute_clip_score(images: List[Image.Image], prompts: List[str], device: str = "cuda") -> Optional[float]:
    try:
        import clip
        model, preprocess = clip.load("ViT-B/32", device=device)
        scores = []
        for img, prompt in zip(images, prompts):
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            text_tokens = clip.tokenize([prompt]).to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img_tensor)
                txt_feat = model.encode_text(text_tokens)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                scores.append((img_feat @ txt_feat.T).item())
        return float(np.mean(scores)) if scores else None
    except Exception:
        return None


class EvaluationReport:

    def __init__(self):
        self.metrics = {}
        self.per_domain = defaultdict(dict)

    def add_metric(self, name: str, value: float, domain: str = None):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return
        if domain:
            self.per_domain[domain][name] = float(value)
        else:
            self.metrics[name] = float(value)

    def add_fid_results(self, fid_global=None, fid_per_domain=None):
        if fid_global is not None:
            self.add_metric("fid_global", fid_global)
        if fid_per_domain:
            for domain, fid in fid_per_domain.items():
                self.add_metric("fid", fid, domain=domain)

    def save_report(self, output_path: Path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"metrics": self.metrics, "per_domain": dict(self.per_domain)}, f, indent=2)
        print(f"Report saved: {output_path}")

    def summary(self) -> str:
        lines = ["=" * 70, "EVALUATION REPORT", "=" * 70]
        for name, value in self.metrics.items():
            lines.append(f"  {name}: {value:.4f}")
        lines.append("=" * 70)
        return "\n".join(lines)

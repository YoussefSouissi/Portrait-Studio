from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import json


def detect_environment():
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        env_type = "kaggle"
        celeba_root = kaggle_input / "datasets" / "jessicali9530" / "celeba-dataset" / "img_align_celeba"
        data_root = kaggle_input / "datasets" / "souissiyoussef" / "diffusion-text2image" / "data" / "processed"
        celeba_img_dir = celeba_root / "img_align_celeba"
        celeba_metadata_dir = celeba_root
        print("Environment: KAGGLE")
    else:
        env_type = "local"
        celeba_root = Path("DATASET/CelebA")
        celeba_img_dir = celeba_root / "img_align_celeba"
        celeba_metadata_dir = celeba_root / "metadata"
        data_root = Path("data/processed")
        print("Environment: LOCAL")
    return env_type, celeba_root, data_root, celeba_img_dir, celeba_metadata_dir


ENVIRONMENT, CELEBA_ROOT, DATA_ROOT, CELEBA_IMG_DIR, CELEBA_METADATA_DIR = detect_environment()


def get_data_root():
    return DATA_ROOT

def get_processed_root():
    return DATA_ROOT

def get_celeba_root():
    return CELEBA_ROOT

def get_celeba_img_dir():
    return CELEBA_IMG_DIR

def get_celeba_metadata_dir():
    return CELEBA_METADATA_DIR

def get_results_root():
    if ENVIRONMENT == "kaggle":
        return Path("/kaggle/working/results")
    return Path(__file__).resolve().parents[1] / "results"

def get_mlflow_root():
    if ENVIRONMENT == "kaggle":
        return Path("/kaggle/working/mlflow")
    return Path(__file__).resolve().parents[1] / "mlflow"


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: int = 16          # ratio alpha/r = 2 — best result from Run 1
    dropout: float = 0.0
    target_modules: list = field(default_factory=lambda: ["to_q", "to_v"])
    init_lora_weights: str = "gaussian"
    use_dora: bool = False
    use_rslora: bool = False

    def to_dict(self):
        return {
            "r": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "init_lora_weights": self.init_lora_weights,
            "use_dora": self.use_dora,
            "use_rslora": self.use_rslora,
        }


@dataclass
class TrainingConfig:
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    vae_model_id: str = "madebyollin/sdxl-vae-fp16-fix"

    lora: LoRAConfig = field(default_factory=LoRAConfig)

    batch_size: int = 1
    gradient_accumulation_steps: int = 4   # effective batch = 4

    @property
    def max_batch_size_total(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    fp16: bool = True
    gradient_checkpointing: bool = True
    use_xformers: bool = False              # disabled on Kaggle T4
    enable_attention_slicing: bool = True
    use_8bit_adam: bool = False             # bitsandbytes absent on Kaggle

    max_train_steps: int = 15000           # best checkpoint: 15000 steps, loss=0.023538
    save_steps: int = 2500
    eval_steps: int = 5000
    logging_steps: int = 100
    max_checkpoints_total: int = 3

    scheduler: str = "constant_with_warmup"
    num_warmup_steps: int = 500

    learning_rate: float = 5e-5
    adam_weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    noise_offset: float = 0.0

    seed: int = 42
    deterministic: bool = True

    output_dir: Optional[str] = None
    logging_dir: Optional[str] = None
    mlflow_dir: Optional[str] = None

    use_mlflow: bool = True
    mlflow_experiment_name: str = "stable-diffusion-lora"
    mlflow_run_name: str = "celeba-sdxl-lora"

    num_workers: int = 2
    pin_memory: bool = True

    use_weighted_sampling: bool = True
    eval_on_train: bool = False

    def resolve_paths(self):
        results_root = get_results_root()
        mlflow_root = get_mlflow_root()
        self.output_dir = str(results_root / "checkpoints")
        self.logging_dir = str(results_root / "logs")
        self.mlflow_dir = str(mlflow_root)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logging_dir).mkdir(parents=True, exist_ok=True)
        Path(self.mlflow_dir).mkdir(parents=True, exist_ok=True)
        return self

    def to_dict(self):
        return {
            "model_id": self.model_id,
            "vae_model_id": self.vae_model_id,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.max_batch_size_total,
            "max_train_steps": self.max_train_steps,
            "learning_rate": self.learning_rate,
            "adam_weight_decay": self.adam_weight_decay,
            "fp16": self.fp16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "use_xformers": self.use_xformers,
            "use_8bit_adam": self.use_8bit_adam,
            "seed": self.seed,
            **self.lora.to_dict(),
            "noise_offset": self.noise_offset,
        }

    def save_to_json(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class EvaluationConfig:
    num_samples_per_domain: int = 100
    num_inference_steps: int = 50
    guidance_scale: float = 9.0
    seed: int = -1                  # -1 = random seed per generation
    batch_size: int = 8
    compute_fid: bool = True
    compute_clip_score: bool = True
    compute_lpips: bool = False
    device: str = "cuda"

    def to_dict(self):
        import dataclasses
        return dataclasses.asdict(self)


TrainConfig = TrainingConfig()
EvalConfig = EvaluationConfig()

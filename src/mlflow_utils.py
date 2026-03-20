import mlflow
import mlflow.pytorch
from pathlib import Path
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MLflowTracker:

    def __init__(self, tracking_uri: Path, experiment_name: str, run_name: str):
        self.tracking_uri = str(tracking_uri)
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.run_id = None

        Path(self.tracking_uri).mkdir(parents=True, exist_ok=True)
        sqlite_path = Path(self.tracking_uri) / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{sqlite_path.as_posix()}")
        mlflow.set_experiment(experiment_name)

    def start_run(self, tags: Optional[Dict[str, str]] = None):
        mlflow.start_run(run_name=self.run_name)
        self.run_id = mlflow.active_run().info.run_id
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)
        logger.info(f"MLflow run started: {self.run_id}")
        return self.run_id

    def log_params(self, params: Dict[str, Any]):
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        for key, value in metrics.items():
            if value is not None:
                mlflow.log_metric(key, float(value), step=step)

    def log_metric(self, key: str, value: float, step: int = None):
        if value is not None:
            mlflow.log_metric(key, float(value), step=step)

    def log_artifact(self, local_path: Path, artifact_path: str = None):
        if Path(local_path).exists():
            mlflow.log_artifact(str(local_path), artifact_path=artifact_path)

    def log_artifacts(self, local_dir: Path, artifact_path: str = None):
        if Path(local_dir).is_dir():
            mlflow.log_artifacts(str(local_dir), artifact_path=artifact_path)

    def log_dict(self, data: Dict[str, Any], filename: str = "data.json"):
        temp_path = Path("/tmp") / filename
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2)
        mlflow.log_artifact(str(temp_path))
        temp_path.unlink()

    def end_run(self, status: str = "FINISHED"):
        mlflow.end_run(status)
        logger.info(f"MLflow run ended: {status}")


class ExperimentTracker:

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history = {
            "steps": [],
            "losses": [],
            "learning_rates": [],
            "timestamps": [],
        }
        self.best_metrics = {}
        self.time_start = datetime.now()

    def log_training_step(self, step: int, loss: float, lr: float = None):
        self.history["steps"].append(step)
        self.history["losses"].append(float(loss))
        if lr is not None:
            self.history["learning_rates"].append(float(lr))
        self.history["timestamps"].append(datetime.now().isoformat())

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        if step is not None:
            self.history.setdefault("eval_steps", []).append(step)
        for key, value in metrics.items():
            self.history.setdefault(key, []).append(float(value) if value is not None else None)

    def update_best_metrics(self, metrics: Dict[str, float]):
        for key, value in metrics.items():
            if value is None:
                continue
            if key not in self.best_metrics or value < self.best_metrics[key]:
                self.best_metrics[key] = float(value)

    def save_history(self, filename: str = "training_history.json"):
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        return path

    def save_best_metrics(self, filename: str = "best_metrics.json"):
        path = self.output_dir / filename
        elapsed = (datetime.now() - self.time_start).total_seconds()
        with open(path, "w") as f:
            json.dump({
                "best_metrics": self.best_metrics,
                "elapsed_seconds": elapsed,
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)
        return path

    def get_summary(self) -> Dict[str, Any]:
        elapsed = (datetime.now() - self.time_start).total_seconds()
        return {
            "total_steps": len(self.history["steps"]),
            "final_loss": self.history["losses"][-1] if self.history["losses"] else None,
            "best_metrics": self.best_metrics,
            "elapsed_seconds": elapsed,
            "start_time": self.time_start.isoformat(),
            "end_time": datetime.now().isoformat(),
        }

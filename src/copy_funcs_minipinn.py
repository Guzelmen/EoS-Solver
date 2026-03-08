import sys
import glob
from pathlib import Path
from types import SimpleNamespace

import wandb
import torch


def find_latest_state_path(pinn_repo_path: str, run_name: str) -> Path:
    weights_dir = Path(pinn_repo_path) / "saving_weights" / run_name
    candidates = sorted(
        weights_dir.glob("weights_epoch_*"),
        key=lambda p: int(p.stem.split("_")[-1])
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {weights_dir}")
    return candidates[-1]


def fetch_config_from_wandb(wandb_run_path: str) -> dict:
    api = wandb.Api()
    run = api.run(wandb_run_path)
    return dict(run.config)


def build_model(pinn_repo_path: str, params, state_dict, device):
    # Save and evict all EoS src entries so PINN's src can take the name
    saved = {k: v for k, v in sys.modules.items()
             if k == "src" or k.startswith("src.")}
    for k in saved:
        del sys.modules[k]

    if pinn_repo_path not in sys.path:
        sys.path.insert(0, pinn_repo_path)

    try:
        from src import models
        mode  = str(params.mode).strip()
        phase = int(params.phase)
        model_class = getattr(models, f"Model_{mode}_phase{phase}")
        model = model_class(params).to(device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    finally:
        # Evict PINN's src from sys.modules
        for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            del sys.modules[k]
        # Remove PINN from front of path
        if pinn_repo_path in sys.path:
            sys.path.remove(pinn_repo_path)
        # Restore our EoS src
        sys.modules.update(saved)
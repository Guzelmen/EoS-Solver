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
    # --- Extract norm stats from checkpoint BEFORE instantiation ---
    # Mirrors miniPINN's infer_model.load_model_from_config_and_state so that
    # params buffers are correct even before load_state_dict overwrites them.
    sd_keys = set(state_dict.keys())
    if "a_mean" in sd_keys and "a_std" in sd_keys:
        params.norm_mode = "standardize"
        try:
            params.standard_mean = float(state_dict["a_mean"].item())
            params.standard_std  = float(state_dict["a_std"].item())
        except Exception as e:
            print(f"[build_model] Warning: could not read a_mean/a_std: {e}")
    elif "a_min" in sd_keys and "a_max" in sd_keys:
        params.norm_mode = "minmax"
    if "T_mean" in sd_keys and "T_std" in sd_keys:
        try:
            params.T_mean = float(state_dict["T_mean"].item())
            params.T_std  = float(state_dict["T_std"].item())
        except Exception as e:
            print(f"[build_model] Warning: could not read T_mean/T_std: {e}")
    if "x_log_mean" in sd_keys and "x_log_std" in sd_keys:
        try:
            params.x_log_mean = float(state_dict["x_log_mean"].item())
            params.x_log_std  = float(state_dict["x_log_std"].item())
        except Exception as e:
            print(f"[build_model] Warning: could not read x_log_mean/x_log_std: {e}")

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
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[build_model] Warning: missing keys in state_dict: {missing}")
        if unexpected:
            print(f"[build_model] Warning: unexpected keys in state_dict: {unexpected}")
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
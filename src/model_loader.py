"""
Model loading and phi(x) inference for the TF-EoS solver.

Mirrors exactly how eval_residual_phase4.py loads the PINN:
  1. fetch_config_from_wandb()  ->  params (SimpleNamespace from wandb run config)
  2. find_latest_state_path()   ->  path to latest .pt in saving_weights/<run_name>/
  3. build_model()              ->  instantiated model with weights loaded

All three functions are imported directly from the PINN repo (eval_residual_phase2
and color_map modules), so the loading logic stays in sync with the training repo
automatically.

The PINN repo and MinimalTFFDintPy are added to sys.path here — this is the only
place in the EoS repo that knows about the sibling repo layout.
"""

import sys
import yaml
import numpy as np
import torch
from pathlib import Path
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_eos_config(config_path: str = None) -> dict:
    """
    Load the EoS solver config yaml.

    Args:
        config_path: path to yaml. Defaults to configs/default.yaml
                     relative to this file's location.

    Returns:
        cfg: dict
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
# sys.path setup
# ---------------------------------------------------------------------------

def _add_repos_to_path(pinn_repo_path: str, fdint_repo_path: str):
    """
    Add the PINN repo and MinimalTFFDintPy to sys.path.
    Safe to call multiple times.
    """
    for p in [pinn_repo_path, fdint_repo_path]:
        if p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# PINN loading  (mirrors eval_residual_phase4.py main())
# ---------------------------------------------------------------------------

def load_pinn(cfg: dict = None, config_path: str = None, device: str = None):
    """
    Load the trained Phase 4 PINN using the same pattern as eval_residual_phase4:
      - fetch config from wandb API
      - find latest checkpoint in saving_weights/<run_name>/
      - build and load model

    Args:
        cfg:         pre-loaded EoS config dict (if None, loads default.yaml)
        config_path: path to EoS config yaml (used only if cfg is None)
        device:      override device string (e.g. "cuda:0"). If None, uses cfg['device'].

    Returns:
        model:  loaded PINN in eval mode, gradients disabled
        params: SimpleNamespace of the wandb run config (mirrors eval script)
    """
    if cfg is None:
        cfg = load_eos_config(config_path)

    _add_repos_to_path(cfg["pinn_repo_path"], cfg["fdint_repo_path"])

    # Import directly from PINN repo modules, exactly as eval_residual_phase4 does
    from src.eval_residual_phase2 import fetch_config_from_wandb, build_model
    from src.color_map import find_latest_state_path

    # Resolve device
    target_device = device or cfg.get("device", "cpu")
    if target_device == "auto":
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(target_device)

    # 1. Fetch config from wandb
    wandb_config = fetch_config_from_wandb(cfg["wandb_run_path"])
    params = SimpleNamespace(**wandb_config)
    params.device = torch_device

    # Ensure debug mode is off for inference
    if not hasattr(params, "debug_mode"):
        params.debug_mode = False

    # Cast numeric fields that wandb may return as strings
    for field, cast in [("x_min_threshold", float), ("random_seed", int)]:
        if hasattr(params, field):
            setattr(params, field, cast(getattr(params, field)))

    # 2. Find latest checkpoint
    state_path = find_latest_state_path(cfg["run_name"])
    print(f"Loading checkpoint: {state_path}")
    state_dict = torch.load(state_path, map_location=torch_device)

    # 3. Build model with weights
    model = build_model(params, state_dict, torch_device)

    torch.set_grad_enabled(False)

    return model, params


# ---------------------------------------------------------------------------
# x grid and phi(x) inference
# ---------------------------------------------------------------------------

def build_x_grid(n_x: int, x_min: float) -> np.ndarray:
    """
    Log-spaced grid from x_min to 1.

    Log spacing clusters points near x=0 where phi(x) varies most rapidly.

    Args:
        n_x:   number of grid points
        x_min: smallest x value

    Returns:
        x: 1D numpy array, shape [n_x]
    """
    return np.logspace(np.log10(x_min), 0.0, n_x)


def predict_phi(
    alpha_1: float,
    T_1_keV: float,
    x_grid: np.ndarray,
    model,
    device: str = "cpu",
) -> np.ndarray:
    """
    Evaluate phi(x; alpha_1, T_1) on the x grid.

    Builds the [x, alpha, T] input tensor expected by the Phase 4 model
    and runs a forward pass.

    Args:
        alpha_1:  Z=1 reduced dimensionless cell radius
        T_1_keV:  Z=1 reduced temperature [keV]
        x_grid:   1D numpy array of x values in (0, 1], shape [n_x]
        model:    loaded PINN (from load_pinn)
        device:   torch device string

    Returns:
        phi: 1D numpy array, shape [n_x]
    """
    n = len(x_grid)
    alpha_col = np.full(n, alpha_1, dtype=np.float32)
    T_col     = np.full(n, T_1_keV, dtype=np.float32)
    inputs_np = np.stack([x_grid.astype(np.float32), alpha_col, T_col], axis=1)

    inputs_t = torch.from_numpy(inputs_np).to(device)

    with torch.no_grad():
        phi_t = model(inputs_t)

    return phi_t.cpu().numpy().reshape(-1)
"""
Model loading and phi(x) inference for the TF-EoS solver.

Mirrors exactly how eval_residual_phase4.py loads the PINN:
  1. fetch_config_from_wandb()  ->  params (SimpleNamespace from wandb run config)
  2. find_state_path_for_epoch() or find_latest_state_path()  ->  checkpoint path
  3. build_model()              ->  instantiated model with weights loaded

The epoch to load is controlled by cfg['epoch']:
  - None (or absent): loads the latest checkpoint via find_latest_state_path()
  - integer:          loads saving_weights/<run_name>/weights_epoch_<epoch>

The PINN repo and MinimalTFFDintPy are added to sys.path here — this is the only
place in the EoS repo that knows about the sibling repo layout.
"""

import sys
import yaml
import numpy as np
import torch
from pathlib import Path
from types import SimpleNamespace

from src.copy_funcs_minipinn import find_latest_state_path, fetch_config_from_wandb, build_model


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

def _add_repos_to_path(fdint_repo_path: str):
    """
    Add the MinimalTFFDintPy to sys.path.
    Safe to call multiple times.
    """
    p = fdint_repo_path
    if p not in sys.path:
        sys.path.append(p)


# ---------------------------------------------------------------------------
# Checkpoint path resolution
# ---------------------------------------------------------------------------

def find_state_path_for_epoch(pinn_repo_path: str, run_name: str, epoch: int) -> Path:
    """
    Return the checkpoint path for a specific training epoch.

    Mirrors the pattern used in the PINN repo's eval_interp_range.py:
        saving_weights/<run_name>/weights_epoch_<epoch>

    Args:
        pinn_repo_path: absolute path to the miniPINN repo root
        run_name:       subfolder name inside saving_weights/
        epoch:          integer epoch number

    Returns:
        Path to the checkpoint file

    Raises:
        FileNotFoundError if the checkpoint does not exist
    """
    weights_dir = Path(pinn_repo_path) / "saving_weights" / run_name
    p = weights_dir / f"weights_epoch_{epoch}"
    if not p.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {p}\n"
            f"Available epochs in {weights_dir}:\n"
            + "\n".join(str(f.name) for f in sorted(weights_dir.glob("weights_epoch_*")))
        )
    return p


# ---------------------------------------------------------------------------
# PINN loading
# ---------------------------------------------------------------------------

def load_pinn(cfg: dict = None, config_path: str = None, device: str = None, verbose: bool = False):
    """
    Load the trained Phase 4 PINN.

    Epoch selection (from cfg['epoch']):
      - None / not present: loads the latest checkpoint (find_latest_state_path)
      - integer:            loads weights_epoch_<epoch> from saving_weights/<run_name>/

    Args:
        cfg:         pre-loaded EoS config dict (if None, loads default.yaml)
        config_path: path to EoS config yaml (used only if cfg is None)
        device:      override device string. If None, uses cfg['device'].

    Returns:
        model:  loaded PINN in eval mode, autograd enabled
        params: SimpleNamespace of the wandb run config
    """
    if cfg is None:
        cfg = load_eos_config(config_path)

    _add_repos_to_path(cfg["fdint_repo_path"])

    # Resolve device
    target_device = device or cfg.get("device", "cpu")
    if target_device == "auto":
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(target_device)

    # 1. Fetch wandb run config
    wandb_config = fetch_config_from_wandb(cfg["wandb_run_path"])
    params = SimpleNamespace(**wandb_config)
    params.device = torch_device

    if not hasattr(params, "debug_mode"):
        params.debug_mode = False

    # Cast numeric fields that wandb may return as strings
    for field, cast in [("x_min_threshold", float), ("random_seed", int)]:
        if hasattr(params, field):
            setattr(params, field, cast(getattr(params, field)))

    # 2. Resolve checkpoint path
    epoch = cfg.get("epoch", None)
    if epoch is not None:
        state_path = find_state_path_for_epoch(
            cfg["pinn_repo_path"], cfg["run_name"], int(epoch)
        )
        print(f"Loading checkpoint at epoch {epoch}: {state_path}")
    else:
        state_path = find_latest_state_path(cfg["pinn_repo_path"], cfg["run_name"])
        print(f"Loading latest checkpoint: {state_path}")

    # 3. Load weights and build model
    state_dict = torch.load(state_path, map_location=torch_device)
    model = build_model(cfg["pinn_repo_path"], params, state_dict, torch_device)

    # Diagnostic: confirm norm-stat buffers loaded from checkpoint
    _loaded = model.state_dict()
    _keys = ["a_mean", "a_std", "T_mean", "T_std", "x_log_mean", "x_log_std"]
    _vals = {k: float(_loaded[k].item()) for k in _keys if k in _loaded}
    _missing = [k for k in _keys if k not in _loaded]
    if verbose:
        print("[load_pinn] Norm-stat buffers after load:")
        for k, v in _vals.items():
            print(f"  {k} = {v:.6g}")
    if _missing:
        print(f"[load_pinn] WARNING: norm buffers absent from model: {_missing}")
    _defaults = {"a_mean": 0.0, "a_std": 1.0, "T_mean": 0.0, "T_std": 1.0,
                 "x_log_mean": 0.0, "x_log_std": 1.0}
    _suspicious = [k for k, v in _vals.items() if abs(v - _defaults.get(k, float("nan"))) < 1e-9]
    if _suspicious:
        print(f"[load_pinn] WARNING: these buffers still hold YAML defaults "
              f"(checkpoint may not contain them): {_suspicious}")

    # Gradients must stay enabled — the hard-constraint BC transform in the
    # model's forward pass uses first_deriv_auto (autograd), so torch.no_grad()
    # would break inference.
    torch.set_grad_enabled(True)

    return model, params


# ---------------------------------------------------------------------------
# x grid
# ---------------------------------------------------------------------------

def build_x_grid(n_x: int, x_min: float) -> np.ndarray:
    """
    Log-spaced grid from x_min to 1.

    Log spacing clusters points near x=0 where phi(x) varies most rapidly.

    Args:
        n_x:   number of grid points
        x_min: smallest x value (avoid true zero, xi -> inf as x -> 0)

    Returns:
        x: 1D numpy array, shape [n_x]
    """
    return np.logspace(np.log10(x_min), np.log10(1.0), n_x)


# ---------------------------------------------------------------------------
# phi(x) inference
# ---------------------------------------------------------------------------

def predict_phi(
    alpha_1: float,
    T_1_keV: float,
    x_grid: np.ndarray,
    model,
    device,
) -> np.ndarray:
    """
    Evaluate phi(x; alpha_1, T_1) on the x grid using the PINN.

    Builds the [N, 3] input tensor [x, alpha, T] expected by the Phase 4
    model and runs a single forward pass.

    Args:
        alpha_1:  Z=1 reduced dimensionless cell radius
        T_1_keV:  Z=1 reduced temperature [keV]
        x_grid:   1D numpy array of x values in (0, 1], shape [n_x]
        model:    loaded PINN (from load_pinn)
        device:   torch device

    Returns:
        phi: 1D numpy array, shape [n_x], values of phi(x)
    """
    n = len(x_grid)
    alpha_col = np.full(n, alpha_1, dtype=np.float32)
    T_col     = np.full(n, T_1_keV, dtype=np.float32)
    inputs_np = np.stack([x_grid.astype(np.float32), alpha_col, T_col], axis=1)

    inputs_t = torch.from_numpy(inputs_np).to(device).requires_grad_(True)

    phi_t = model(inputs_t)

    return phi_t.detach().cpu().numpy().reshape(-1)
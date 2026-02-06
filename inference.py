# ======================================================
# ===================== IMPORTS ========================
# ======================================================
import torch
import numpy as np
import pandas as pd
import os
import pickle
from pathlib import Path
from torch.utils.data import DataLoader

import Pipeline_2 as Pipeline

from models.LSTM import LSTM_two_layers
from models.GRU import GRU_two_layers
from models.LSTM_FCN import LSTM_FCN

from utils.graph_pipeline import (
    plot_continuous_horizon0,
    plot_one_day,
    plot_scatter_real_vs_pred,
)

# ======================================================
# ================= MODEL FACTORY ======================
# ======================================================
MODEL_FACTORY = {
    "LSTM": LSTM_two_layers,
    "GRU": GRU_two_layers,
    "LSTM_FCN": LSTM_FCN,
}

# ======================================================
# ================= LOAD TRAINED MODEL ================
# ======================================================
def load_trained_model(checkpoint_path, device, input_size):
    # 1. Existence verification
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ Model not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 2. Identify keys (compatibility with old and new versions)
    state_dict_key = "state_dict" if "state_dict" in checkpoint else "model_state_dict"
    if state_dict_key not in checkpoint:
        # If no key dictionary, maybe the file is just the state_dict directly
        state_dict = checkpoint
        model_name = "GRU" if "GRU" in checkpoint_path else "LSTM_FCN"
        print(f"⚠️ Simple checkpoint detected. Assuming {model_name} based on filename.")
    else:
        state_dict = checkpoint[state_dict_key]
        model_name = checkpoint.get("model_name", "LSTM_FCN")

    # 3. Configuration Handling (Avoids ASSERT error)
    if "config" in checkpoint:
        cfg = checkpoint["config"]["model"]
        full_cfg = checkpoint["config"]
    else:
        print("⚠️ Checkpoint does not have 'config'. Using emergency standard values.")
        # Default values so it doesn't stop the first time
        cfg = {
            "hidden_size": 128, # This value will be corrected later if it's LSTM_FCN
            "output_size": 24, # or output_window
            "output_window": 24,
            "dropout": 0.3,
            "length": 24,
            "lag": 0
        }
        full_cfg = {"model": cfg}

    # 4. Reconstruct model parameters
    if model_name in ["LSTM", "GRU"]:
        model_params = {
            "input_size": input_size,
            "hidden_size": cfg.get("hidden_size", 128),
            "output_size": cfg.get("output_size", 24),
            "dropout": cfg.get("dropout", 0.3),
        }
    elif model_name == "LSTM_FCN":
        model_params = {
            "input_size": input_size,
            "hidden_size": cfg.get("hidden_size", 128),
            "output_window": cfg.get("output_window", 24),
            "dropout": cfg.get("dropout", 0.3),
        }

    # 5. Load weights
    model_class = MODEL_FACTORY[model_name]
    model = model_class(**model_params)
    
    # load_state_dict with strict=False in case of minor variations
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    print(f"\n✅ Checkpoint loaded successfully.")
    return model, model_name, model_params, full_cfg

# ======================================================
# =================== INFERENCE ========================
# ======================================================
def inference_model(model, dataloader, device):
    preds, targets = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            out = model(x).cpu().numpy()
            # Ensure shape is (Batch, Window)
            if out.ndim == 3: out = out.squeeze(-1)
            preds.append(out)
            
            y_np = y.numpy()
            if y_np.ndim == 3: y_np = y_np.squeeze(-1)
            targets.append(y_np)

    return (
        np.concatenate(preds, axis=0),
        np.concatenate(targets, axis=0),
    )

# ======================================================
# =================== RUN ==============================
# ======================================================
def run_inference(best_model_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    
    # --- PATH AUTO-DETECTION ---
    # We look for the most recent .pt or .pth file in root or checkpoints/
    import os

    CHECKPOINT_PATH = best_model_path  # If a specific path is provided, use it

    if CHECKPOINT_PATH is None:
        print("❌ No model file found in Drive. Check paths.")

    DATA_PATH = "./data/Processed"
    inf_x, inf_y = Pipeline.load_split("inference", DATA_PATH)
    input_size = inf_x.shape[1]
    print(f"📊 Inference: Detected {input_size} input variables.")

    # --- LOAD MODEL ---
    model, model_name, model_params, full_cfg = load_trained_model(
        CHECKPOINT_PATH,
        device,
        input_size = input_size,
    )

    # --- LOAD DATA ---
    ds_inf = Pipeline.TimeSeriesDataset(
        inf_x,
        inf_y,
        length=full_cfg["model"]["length"],
        lag= full_cfg["model"]["lag"],
        output_window=full_cfg["model"]["output_window"],
        stride=1,
    )

    dl_inf = DataLoader(ds_inf, batch_size=64, shuffle=False)

    # --- INFERENCE ---
    preds, targets = inference_model(model, dl_inf, device)

    # --- DE-NORMALIZE ---
    with open(f"{DATA_PATH}/stats.pkl", "rb") as f:
        stats = pickle.load(f)

    y_mu = stats["Y"]["mu"]["Energy"]
    y_std = stats["Y"]["std"]["Energy"]

    preds_real = preds * y_std + y_mu
    targets_real = targets * y_std + y_mu

    # --- METRICS ---
    mase_scale = stats["mase"]["scale"]
    mase_h0 = np.mean(np.abs(preds_real[:, 0] - targets_real[:, 0])) / mase_scale
    rmse_h0 = np.sqrt(np.mean((preds_real[:, 0] - targets_real[:, 0]) ** 2))

    print("\n========== INFERENCE METRICS ==========")
    print(f"MASE horizon=0: {mase_h0:.4f}")
    print(f"RMSE horizon=0: {rmse_h0:.4f} kWh")
    
    # --- PLOTS ---
    # Take a 24-hour slice to represent one day
    day_real = targets_real[0:24].flatten()
    day_pred = preds_real[0:24].flatten()
    
    if len(day_real) == 24:
        plot_one_day(day_real, day_pred, day_idx=10)
    else:
        print("Not enough data to plot 24h")
    
    plot_continuous_horizon0(targets_real, preds_real, start_idx=0, n_days=7)
    plot_scatter_real_vs_pred(targets_real, preds_real)

    # --- SAVE OUTPUT ---
    df = pd.DataFrame(preds_real, columns=[f"h_{i}" for i in range(preds_real.shape[1])])
    out_path = "./outputs/inference_predictions.xlsx"
    os.makedirs("./outputs", exist_ok=True)
    df.to_excel(out_path, index=False)

    print(f"\n✅ Inference saved to: {out_path}")

if __name__ == "__main__":
    # Note: Ensure Pipeline.training() is also renamed in the imported script
    best_model_path = Pipeline.main() 
    run_inference(best_model_path=best_model_path)
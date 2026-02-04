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
    # 1. Verificación de existencia
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ No se encontró el modelo en: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 2. Identificar claves (compatibilidad con versiones viejas y nuevas)
    state_dict_key = "state_dict" if "state_dict" in checkpoint else "model_state_dict"
    if state_dict_key not in checkpoint:
        # Si no hay diccionario de claves, quizás el archivo es solo el state_dict directamente
        state_dict = checkpoint
        model_name = "GRU" if "GRU" in checkpoint_path else "LSTM_FCN"
        print(f"⚠️ Checkpoint simple detectado. Asumiendo {model_name} por nombre de archivo.")
    else:
        state_dict = checkpoint[state_dict_key]
        model_name = checkpoint.get("model_name", "LSTM_FCN")

    # 3. Manejo de la Configuración (EVITA EL ERROR DE ASSERT)
    if "config" in checkpoint:
        cfg = checkpoint["config"]["model"]
        full_cfg = checkpoint["config"]
    else:
        print("⚠️ El checkpoint no tiene 'config'. Usando valores estándar de emergencia.")
        # Valores por defecto para que no se detenga la primera vez
        cfg = {
            "hidden_size": 128, # Este valor se corregirá más adelante si es LSTM_FCN
            "output_size": 24, # o output_window
            "output_window": 24,
            "dropout": 0.3,
            "length": 24,
            "lag": 0
        }
        full_cfg = {"model": cfg}

    # 4. Reconstruir parámetros del modelo
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

    # 5. Cargar pesos
    model_class = MODEL_FACTORY[model_name]
    model = model_class(**model_params)
    
    # load_state_dict con strict=False por si hay pequeñas variaciones
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    print(f"\n✅ Checkpoint cargado con éxito.")
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
            # Asegurar que la forma sea (Batch, Window)
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
    
    # --- AUTO-DETECCIÓN DE PATHS ---
    # Buscamos el archivo .pt o .pth más reciente en la raíz o en checkpoints/
    # Crea una carpeta llamada 'Proyecto_IA' en tu Google Drive manualmente o por código
    import os

    CHECKPOINT_PATH = best_model_path  # Si se proporciona un path específico, úsalo

    if CHECKPOINT_PATH is None:
        print("❌ No se encontró ningún archivo de modelo en Drive. Revisa las rutas.")

    DATA_PATH = "./data/Processed"
    inf_x, inf_y = Pipeline.load_split("inference", DATA_PATH)
    input_size = inf_x.shape[1]
    print(f"📊 Inferencia: Detectadas {input_size} variables de entrada.")

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

    # --- DESNORMALIZAR ---
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
    
    # --- PLOTS CORREGIDOS ---
    # En lugar de [0], tomamos una tajada de 24 horas para representar un día
    dia_real = targets_real[0:24].flatten()
    dia_pred = preds_real[0:24].flatten()
    
    if len(dia_real) == 24:
        plot_one_day(dia_real, dia_pred, day_idx=10)
    else:
        print("No hay suficientes datos para graficar 24h")
    
    plot_continuous_horizon0(targets_real, preds_real, start_idx=0, n_days=7)
    plot_scatter_real_vs_pred(targets_real, preds_real)

    # --- SAVE OUTPUT ---
    df = pd.DataFrame(preds_real, columns=[f"h_{i}" for i in range(preds_real.shape[1])])
    out_path = "./outputs/inference_predictions.xlsx"
    os.makedirs("./outputs", exist_ok=True)
    df.to_excel(out_path, index=False)

    print(f"\n✅ Inferencia guardada en: {out_path}")

if __name__ == "__main__":
    best_model_path = Pipeline.training()
    run_inference(best_model_path=best_model_path)
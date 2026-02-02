import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# Intentar importar dependencias de archivos locales
try:
    from models.LSTM import LSTM_two_layers
    from utils.metrics import mase
    from utils.graph_pipeline import plot_continuous_horizon0, plot_one_day
except ImportError:
    print("Nota: Asegúrate de tener los módulos 'models' y 'utils' en tu directorio.")

# ======================================================
# 1. DATASET Y PROCESAMIENTO
# ======================================================

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, length, lag, output_window, stride=1):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        if self.y.ndim == 1: self.y = self.y.unsqueeze(-1)

        self.length, self.lag, self.output_window = length, lag, output_window
        
        N = len(X)
        t0_min = lag + length
        t0_max = N - output_window
        self.starts = np.arange(t0_min, t0_max + 1, stride)

    def __len__(self): return len(self.starts)

    def __getitem__(self, idx):
        t0 = self.starts[idx]
        x = self.X[t0 - self.lag - self.length : t0 - self.lag]
        y = self.y[t0 : t0 + self.output_window]
        return x, y

# ======================================================
# 2. FUNCIONES DE MOTOR (TRAIN, EVAL, INF)
# ======================================================

def train_engine(model, dataloader, epochs, lr, device):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.HuberLoss() # Robusto ante outliers
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds.squeeze(), y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.5f}")

def evaluate_engine(model, dataloader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in dataloader:
            out = model(x.to(device)).detach().cpu().numpy()
            # Elimina dimensiones de tamaño 1 (ej: de [1, 24, 1] a [1, 24])
            preds.append(out.squeeze())
            targets.append(y.numpy().squeeze())
    return np.concatenate(preds), np.concatenate(targets)

# ======================================================
# 3. PIPELINE DE DATOS (TU LÓGICA DE EXCEL)
# ======================================================

def load_and_preprocess_all(ruta_x, ruta_y):
    # Carga simplificada para el ejemplo
    x_df = pd.concat([pd.read_excel(ruta_x, sheet_name=0), pd.read_excel(ruta_x, sheet_name=1)])
    y_df = pd.concat([pd.read_excel(ruta_y, sheet_name=0), pd.read_excel(ruta_y, sheet_name=1)])
    
    # Aquí irían tus funciones: preparar_x_df, preparar_y_df, dividir_x_df, etc.
    # Por brevedad, asumimos que devuelven arrays listos tras aplicar el split temporal
    # [ESTA PARTE USA TUS FUNCIONES DEL PIPELINE]
    return x_df, y_df 

# ======================================================
# 4. EJECUCIÓN PRINCIPAL (MAIN)
# ======================================================

def main():
    # --- CONFIGURACIÓN ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR = Path("./data/Processed")
    
    # Hiperparámetros (puedes cargarlos de tu yaml)
    cfg = {
        "length": 72, "lag": 0, "output_window": 24, "batch_size": 32,
        "epochs": 50, "learning_rate": 1e-3, "hidden_size": 128, "dropout": 0.2
    }

    # --- 1. PREPARAR DATOS ---
    # Asumimos que ya corriste tu pipeline_training() y existen los archivos .xlsx
    train_df = pd.read_excel(OUT_DIR / "train.xlsx", index_col=0)
    val_df   = pd.read_excel(OUT_DIR / "val.xlsx", index_col=0)
    
    train_x = train_df.drop(columns=["Energy"]).to_numpy(dtype=np.float32)
    train_y = train_df["Energy"].to_numpy(dtype=np.float32)
    val_x   = val_df.drop(columns=["Energy"]).to_numpy(dtype=np.float32)
    val_y   = val_df["Energy"].to_numpy(dtype=np.float32)

    ds_train = TimeSeriesDataset(train_x, train_y, cfg["length"], cfg["lag"], cfg["output_window"])
    dl_train = DataLoader(ds_train, batch_size=cfg["batch_size"], shuffle=True)
    
    ds_val = TimeSeriesDataset(val_x, val_y, cfg["length"], cfg["lag"], cfg["output_window"], stride=24)
    dl_val = DataLoader(ds_val, batch_size=cfg["batch_size"], shuffle=False)

    # --- 2. ENTRENAMIENTO ---
    input_size = train_x.shape[1]
    model = LSTM_two_layers(input_size, cfg["hidden_size"], cfg["output_window"], cfg["dropout"])
    
    print(f"Entrenando en {device}...")
    train_engine(model, dl_train, cfg["epochs"], cfg["learning_rate"], device)

    # --- 3. EVALUACIÓN Y GUARDADO ---
    p_val, t_val = evaluate_engine(model, dl_val, device)
    
    # Invertir logaritmo para métricas reales
    p_real, t_real = np.expm1(p_val), np.expm1(t_val)
    train_y_real = np.expm1(train_y)
    
    m = mase(t_real[:, 0], p_real[:, 0], train_y_real)
    print(f"Validación MASE (h=0): {m:.4f}")

    torch.save({"state_dict": model.state_dict(), "cfg": cfg}, "best_model.pth")

    # --- 4. INFERENCIA (HOJA 2 / DATOS NUEVOS) ---
    print("\n--- Iniciando Inferencia sobre Datos Nuevos (Hoja 2) ---")
    inf_df = pd.read_excel(OUT_DIR / "inference.xlsx", index_col=0)
    inf_x = inf_df.drop(columns=["Energy"]).to_numpy(dtype=np.float32)
    inf_y = inf_df["Energy"].to_numpy(dtype=np.float32)

    ds_inf = TimeSeriesDataset(inf_x, inf_y, cfg["length"], cfg["lag"], cfg["output_window"], stride=1)
    dl_inf = DataLoader(ds_inf, batch_size=1, shuffle=False)

    p_inf, t_inf = evaluate_engine(model, dl_inf, device)
    p_inf_real, t_inf_real = np.expm1(p_inf), np.expm1(t_inf)

    target_plot = t_inf_real.flatten()[:24]
    pred_plot = p_inf_real.flatten()[:24]

    # Verificación de seguridad antes de graficar
    if len(target_plot) == 24 and len(pred_plot) == 24:
        plot_one_day(target_plot, pred_plot, day_idx=0)
        print("Inferencia completada y gráfico generado con éxito.")
    else:
        print(f"Error: Dimensiones incorrectas. Target: {target_plot.shape}, Pred: {pred_plot.shape}")

if __name__ == "__main__":
    main()
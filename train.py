import yaml
import torch
import numpy as np
import pandas as pd
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# Nota: Asegúrate de que estas rutas sean correctas en tu entorno de Colab
# o que los archivos estén en la misma carpeta.
try:
    from models.LSTM import LSTM_two_layers
    from models.GRU import GRU_two_layers
    from models.LSTM_FCN import LSTM_FCN
    from models.Transformer import TransformerForecast
    from utils.metrics import rmse, mase
except ImportError:
    print("Asegúrate de subir las carpetas 'models' y 'utils' a tu sesión de Colab.")

# --- DATASET CORREGIDO ---
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, length, lag, output_window, stride=1):
        # Convertimos a tensores en el init para velocidad, pero manejamos la memoria
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        # Asegurar que y tenga la forma correcta (muestras, ventana)
        if self.y.ndim == 1:
            self.y = self.y.unsqueeze(-1)

        N = len(X)
        t0_min = lag + length
        t0_max = N - output_window
        self.starts = np.arange(t0_min, t0_max + 1, stride)

        self.length = length
        self.lag = lag
        self.output_window = output_window

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        t0 = self.starts[idx]
        x = self.X[t0 - self.lag - self.length : t0 - self.lag]
        y = self.y[t0 : t0 + self.output_window]
        return x, y

# --- ENTRENAMIENTO MEJORADO ---
def train_model(model, dataloader, epochs, lr, device):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # AdamW es más estable
    loss_fn = nn.HuberLoss() # Mejor que L1 para evitar saltos bruscos en el gradiente

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            preds = model(x)
            
            # Ajuste de dimensiones si el modelo devuelve (batch, seq, 1) y y es (batch, seq)
            if preds.shape != y.shape:
                preds = preds.squeeze(-1) if preds.shape[-1] == 1 else preds
            
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.6f}")

# --- EVALUACIÓN SIN ERRORES DE MEMORIA ---
def evaluate(model, dataloader, device):
    model.eval()
    preds_list, targets_list = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            # Pasamos a CPU y numpy inmediatamente para no llenar la GPU
            out = model(x).detach().cpu().numpy()
            preds_list.append(out)
            targets_list.append(y.numpy())

    return np.concatenate(preds_list, axis=0), np.concatenate(targets_list, axis=0)

def load_split(name, base):
    # En Colab, asegúrate de que la ruta exista
    path = Path(base) / f"{name}.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
        
    df = pd.read_excel(path, index_col=0)
    X_df = df.drop(columns=["Energy"])
    y = df["Energy"].to_numpy(dtype=np.float32)
    X = X_df.to_numpy(dtype=np.float32)
    feature_columns = X_df.columns.tolist()

    assert "Energy" not in feature_columns, "Energy leaked into features"
    return X, y, feature_columns

# --- FLUJO PRINCIPAL ---
def main():
    # 1. Configuración (Simulada si no tienes el .yaml a mano)
    # Si tienes el yaml, descomenta las líneas de abajo:
    # with open("config/timeseries.yaml") as f:
    #     cfg = yaml.safe_load(f)["model"]
    
    cfg = {
        "length": 168, "lag": 0, "output_window": 24, "batch_size": 32,
        "epochs": 50, "learning_rate": 1e-3, "hidden_size": 64, 
        "output_size": 24, "dropout": 0.1
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando: {device}")
    
    # Rutas de datos (Ajusta a '/content/...' si usas Google Drive o carga local en Colab)
    data_path = "data/Processed"
    
    train_x, train_y, feature_columns = load_split("train", data_path)
    val_x, val_y, _ = load_split("val", data_path)
    input_size = train_x.shape[1]

    # 2. Datasets y Loaders
    ds_train = TimeSeriesDataset(train_x, train_y, cfg["length"], cfg["lag"], cfg["output_window"])
    ds_val = TimeSeriesDataset(val_x, val_y, cfg["length"], cfg["lag"], cfg["output_window"], stride=24)

    dl_train = DataLoader(ds_train, batch_size=cfg["batch_size"], shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=cfg["batch_size"], shuffle=False)

    # 3. Modelos
    models_dict = {
        "LSTM": LSTM_two_layers(input_size, cfg["hidden_size"], cfg["output_size"], cfg["dropout"]),
        "GRU": GRU_two_layers(input_size, cfg["hidden_size"], cfg["output_size"], cfg["dropout"]),
        "LSTM_FCN": LSTM_FCN(input_size, cfg["hidden_size"], cfg["output_window"], cfg["dropout"]),
        "Transformer": TransformerForecast(input_size, 128, 8, 4, 256, cfg["dropout"], cfg["output_window"]),
    }

    best_mase = np.inf
    best_model_name = None

    # 4. Loop de entrenamiento y selección
    for name, model in models_dict.items():
        print(f"\n--- Entrenando Modelo: {name} ---")
        train_model(model, dl_train, cfg["epochs"], cfg["learning_rate"], device)
        
        preds, targets = evaluate(model, dl_val, device)

        # CORRECCIÓN DE EVALUACIÓN:
        # Revertimos logaritmo (expm1) y evaluamos toda la ventana si es posible
        # Si p y t son (N, 24), calculamos sobre el promedio o el primer punto
        p_orig = np.expm1(preds)
        t_orig = np.expm1(targets)
        train_y_orig = np.expm1(train_y)

        # Calculamos MASE sobre la primera predicción de la ventana (típico en forecasting)
        # o puedes aplanar ambos para un MASE global.
        m = mase(t_orig[:, 0], p_orig[:, 0], train_y_orig)

        print(f"Resultado {name} - MASE: {m:.4f}")

        if m < best_mase:
            best_mase = m
            best_model_name = name
            torch.save({
                "model_name": name,
                "state_dict": model.state_dict(),
                "feature_columns": feature_columns,
                "best_mase": best_mase,
            }, "best_model.pth")

    print(f"\n🏆 Proceso terminado. Mejor modelo: {best_model_name} con MASE: {best_mase:.4f}")

if __name__ == "__main__":
    main()
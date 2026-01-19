import torch
import numpy as np
import yaml
import pandas as pd

from torch.utils.data import DataLoader
from train import TimeSeriesDataset
from models.LSTM import LSTM_two_layers
from models.GRU import GRU_two_layers
from models.LSTM_FCN import LSTM_FCN
from models.Transformer import TransformerForecast

from utils.plots import (
    plot_continuous_horizon0,
    plot_one_day,
    plot_scatter_real_vs_pred,
)


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------
    # Config
    # --------------------------------------------------
    with open("config/timeseries.yaml") as f:
        cfg = yaml.safe_load(f)["model"]

    # --------------------------------------------------
    # Load checkpoint (PyTorch 2.6+ compatible)
    # --------------------------------------------------
    checkpoint = torch.load(
        "best_model.pth",
        map_location=device,
        weights_only=False,
    )

    model_name = checkpoint["model_name"]
    train_cols = checkpoint["feature_columns"]
    input_size = checkpoint["input_size"]

    # --------------------------------------------------
    # Load inference data as DataFrame
    # --------------------------------------------------
    df = pd.read_excel("data/Processed/inference.xlsx", index_col=0)

    y = df["Energy"].to_numpy(dtype=np.float32)
    X_df = df.drop(columns=["Energy"])

    # --------------------------------------------------
    # FEATURE COMPARISON (DEBUG / EXPLANATION)
    # --------------------------------------------------
    train_cols_set = set(train_cols)
    inf_cols_set = set(X_df.columns)

    missing_in_inf = train_cols_set - inf_cols_set
    extra_in_inf = inf_cols_set - train_cols_set

    print("\n================ FEATURE COMPARISON ================")
    print(f"Train features ({len(train_cols_set)}):")
    print(sorted(train_cols_set))

    print(f"\nInference features ({len(inf_cols_set)}):")
    print(sorted(inf_cols_set))

    print(f"\nMissing in inference → added as 0 ({len(missing_in_inf)}):")
    print(sorted(missing_in_inf))

    print(f"\nExtra in inference → removed ({len(extra_in_inf)}):")
    print(sorted(extra_in_inf))
    print("===================================================\n")

    # --------------------------------------------------
    # FORCE FEATURE ALIGNMENT (BEST PRACTICE)
    # --------------------------------------------------
    # - elimina columnas extra
    # - añade columnas faltantes con 0
    # - mantiene el orden del training
    X_df = X_df.reindex(
        columns=train_cols,
        fill_value=0.0
    )

    # HARD SAFETY CHECK
    assert X_df.shape[1] == input_size, (
        f"Feature mismatch after alignment: "
        f"{X_df.shape[1]} vs {input_size}"
    )

    # --------------------------------------------------
    # Convert to NumPy (ONLY NOW)
    # --------------------------------------------------
    X = X_df.to_numpy(dtype=np.float32)

    # --------------------------------------------------
    # Build model (MUST match training)
    # --------------------------------------------------
    model = {
        "LSTM": LSTM_two_layers,
        "GRU": GRU_two_layers,
        "LSTM_FCN": LSTM_FCN,
        "Transformer": TransformerForecast,
    }[model_name](
        input_size,
        cfg["hidden_size"],
        cfg["output_window"],
        cfg["dropout"],
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    # --------------------------------------------------
    # Dataset & DataLoader
    # --------------------------------------------------
    ds = TimeSeriesDataset(
        X,
        y,
        cfg["length"],
        cfg["lag"],
        cfg["output_window"],
        stride=24,
    )

    dl = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
    )

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    preds, targets = [], []

    with torch.no_grad():
        for x, t in dl:
            preds.append(model(x.to(device)).cpu().numpy())
            targets.append(t.numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)


    # --------------------------------------------------
    # Plots
    # --------------------------------------------------
    plot_continuous_horizon0(targets, preds)
    plot_one_day(targets, preds,0)
    plot_scatter_real_vs_pred(targets, preds)
    print("Preds log min/max:", preds.min(), preds.max())
    print("Targets log min/max:", targets.min(), targets.max())


if __name__ == "__main__":
    main()

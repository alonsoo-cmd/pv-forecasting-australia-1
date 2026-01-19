import yaml
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from models.LSTM import LSTM_two_layers
from models.GRU import GRU_two_layers
from models.LSTM_FCN import LSTM_FCN
from models.Transformer import TransformerForecast

from utils.metrics import rmse, mase



class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, length, lag, output_window, stride=1):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        if self.y.ndim == 2:
            self.y = self.y.squeeze(-1)

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


def train_model(model, dataloader, epochs, lr, device):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    for _ in range(epochs):
        model.train()
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()


def evaluate(model, dataloader, device):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for x, y in dataloader:
            preds.append(model(x.to(device)).cpu().numpy())
            targets.append(y.numpy())

    return np.concatenate(preds), np.concatenate(targets)


def load_split(name, base):
    df = pd.read_excel(Path(base) / f"{name}.xlsx", index_col=0)

    X_df = df.drop(columns=["Energy"])
    y = df["Energy"].to_numpy(dtype=np.float32)
    X = X_df.to_numpy(dtype=np.float32)

    feature_columns = X_df.columns.tolist()

    # SAFETY CHECK (MUY IMPORTANTE)
    assert "Energy" not in feature_columns, "Energy leaked into feature_columns"

    return X, y, feature_columns



def main():
    with open("config/timeseries.yaml") as f:
        cfg = yaml.safe_load(f)["model"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_x, train_y, feature_columns = load_split("train", "data/Processed")
    val_x, val_y, feature_columns = load_split("val", "data/Processed")
    input_size = train_x.shape[1]

    ds_train = TimeSeriesDataset(train_x, train_y, cfg["length"], cfg["lag"], cfg["output_window"])
    ds_val = TimeSeriesDataset(val_x, val_y, cfg["length"], cfg["lag"], cfg["output_window"], stride=24)

    dl_train = DataLoader(ds_train, batch_size=cfg["batch_size"], shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=cfg["batch_size"], shuffle=False)

    models = {
        "LSTM": LSTM_two_layers(input_size, cfg["hidden_size"], cfg["output_size"], cfg["dropout"]),
        "GRU": GRU_two_layers(input_size, cfg["hidden_size"], cfg["output_size"], cfg["dropout"]),
        "LSTM_FCN": LSTM_FCN(input_size, cfg["hidden_size"], cfg["output_window"], cfg["dropout"]),
        "Transformer": TransformerForecast(input_size, 128, 8, 4, 256, cfg["dropout"], cfg["output_window"]),
    }

    best_mase = np.inf

    for name, model in models.items():
        print(f"Training: {name}")
        train_model(model, dl_train, cfg["epochs"], cfg["learning_rate"], device)
        p, t = evaluate(model, dl_val, device)

        p, t = np.expm1(p[:, 0]), np.expm1(t[:, 0])
        m = mase(t, p, np.expm1(train_y))

        if m < best_mase:
            best_mase = m
            torch.save(
                {
                    "model_name": name,
                    "state_dict": model.state_dict(),
                    "feature_columns": feature_columns,
                    "input_size": len(feature_columns),
                    "best_mase": best_mase,
                },
                "best_model.pth",
            )

            best_model = name


    print("Best model saved", best_model)


if __name__ == "__main__":
    main()

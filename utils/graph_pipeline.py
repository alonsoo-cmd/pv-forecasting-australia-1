
import numpy as np
import matplotlib.pyplot as plt

def plot_continuous_horizon0(
    y_true,
    y_pred,
    start_idx=0,
    n_days=7,
    title=None
):
    """
    Serie continua usando SOLO horizon=0:
    y[t] vs yhat[t] donde yhat[t] = preds[window_t][0]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim == 2:
        y_true = y_true[:, 0]
    if y_pred.ndim == 2:
        y_pred = y_pred[:, 0]

    n_hours = n_days * 24
    end = start_idx + n_hours

    y_true = y_true[start_idx:end]
    y_pred = y_pred[start_idx:end]

    x = np.arange(len(y_true))

    plt.figure(figsize=(14, 4))
    plt.plot(x, y_true, label="Real", linewidth=2)
    plt.plot(x, y_pred, "--", label="Predicho", linewidth=2)

    plt.xlabel("Horas")
    plt.ylabel("Energía (kWh)")
    plt.title(title or f"Predicción continua horizon=0 ({n_days} días)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_one_day(y_true, y_pred, day_idx):
    yt = y_true[day_idx]
    yp = y_pred[day_idx]
    err = yp - yt

    x = np.arange(24)

    plt.figure(figsize=(12, 4))
    plt.plot(x, yt, label="Real", linewidth=2)
    plt.plot(x, yp, "--", label="Predicho", linewidth=2)
    plt.title(f"Día {day_idx} – Real vs Predicho")
    plt.xlabel("Hora del día")
    plt.ylabel("Energía (kWh)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 3))
    plt.bar(x, err)
    plt.axhline(0, color="black")
    plt.title("Error por hora")
    plt.xlabel("Hora")
    plt.ylabel("Error (kWh)")
    plt.tight_layout()
    plt.show()

def plot_scatter_real_vs_pred(y_true, y_pred):
    y_true = y_true[:, 0]
    y_pred = y_pred[:, 0]

    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.3)
    maxv = max(y_true.max(), y_pred.max())
    plt.plot([0, maxv], [0, maxv], "r--")
    plt.xlabel("Real (kWh)")
    plt.ylabel("Predicho (kWh)")
    plt.title("Real vs Predicho (horizon=0)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# PV Power Forecasting with Deep Learning (Australia)

This repository contains an end-to-end, reproducible workflow for photovoltaic (PV) power forecasting using PyTorch. The goal is to train and compare multiple deep learning models on time-series data, select the best one using consistent metrics (including MASE), and then run inference to generate predictions, plots, and an exportable output file.

The workflow is designed to be straightforward:

* prepare and normalize data (no leakage)
* train and validate several architectures under the same protocol
* save the best checkpoint
* run inference + evaluation on a dedicated split

---

## Project layout

```text
project/
│
├── data\_pipeline.py            # Raw -> Processed (feature engineering + normalization + stats)
├── Pipeline\_2.py               # Training + validation + best-model selection (checkpoint saving)
├── inference.py                # Load best checkpoint -> inference -> metrics -> plots -> Excel export
│
├── config/
│   └── timeseries.yaml         # Hyperparameters and dataset settings
│
├── models/
│   ├── LSTM.py
│   ├── GRU.py
│   └── LSTM\_FCN.py
│
├── utils/
│   └── graph\_pipeline.py       # Plot utilities used by training/inference
│
├── data/
│   ├── Raw/                    # Original Excel files (usually not committed)
│   └── Processed/
│       ├── train.xlsx
│       ├── val.xlsx
│       ├── test.xlsx
│       ├── inference.xlsx
│       └── stats.pkl
│
└── outputs/
└── inference\_predictions.xlsx
```

---

## Setup

Create a virtual environment (optional) and install dependencies:

```bash
pip install -r requirements.txt
```

Note: the pipeline runs well in Google Colab too (GPU is optional).

---

## Data preparation (Raw -> Processed)

Raw inputs are expected as Excel files:

* `data/Raw/pv\_dataset\_full.xlsx` (PV target series)
* `data/Raw/wx\_dataset\_full.xlsx` (weather/features)

Run the preprocessing script to generate normalized splits and inference inputs:

```bash
python data\_pipeline.py
```

This produces:

* `data/Processed/train.xlsx`, `val.xlsx`, `test.xlsx`
* `data/Processed/inference.xlsx`
* `data/Processed/stats.pkl` with:

  * normalization parameters for X and Y (computed on train only)
  * MASE scale (computed on the real series with daily seasonality, m=24)

x Important: normalization is done without leakage (mean/std are fitted on the training portion only).

---

## Training and model selection

Training evaluates multiple architectures using the same dataset splits and evaluation logic, then selects the best model based on validation performance (MAE at horizon 0 in kWh).

```bash
python Pipeline\_2.py
```

What happens:

* loads `train/val/test` from `data/Processed`
* trains and validates: LSTM, GRU, LSTM\_FCN
* keeps the best model by validation MAE (horizon 0)
* saves a checkpoint (by default configured for Colab/Drive paths)

Output checkpoint name:

* `best\_model\_<MODEL>.pt`

If you are not running in Colab, update the checkpoint directory in `Pipeline\_2.py` (the default path points to Drive).

---

## Inference and evaluation

Inference loads the selected checkpoint, runs predictions on the inference split, computes metrics, generates plots, and exports an Excel file with the predicted horizons.



```bash
python inference.py
```

Outputs:

* Console metrics at horizon 0:

  * MASE
  * RMSE (kWh)

* Plots:

  * 24h example day
  * continuous horizon-0 comparison across multiple days
  * scatter plot (real vs predicted)

* File:

  * `outputs/inference\_predictions.xlsx`

---

## Models

* LSTM (two layers)
* GRU (two layers)
* LSTM-FCN

All models are implemented in `models/` and trained under the same dataset protocol (lookback length, lag, output window).

---

## Notes (implementation details)

* Feature engineering includes time-based cyclic encodings (hour/month/weekday) and solar position features (elevation/azimuth, plus circular encoding) computed via `pvlib`.
* Evaluation uses consistent scaling and de-normalization via `data/Processed/stats.pkl`.
* Log transforms (`log1p` / `expm1`) are used where appropriate during evaluation.

---


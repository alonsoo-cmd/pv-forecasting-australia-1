import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from pvlib.solarposition import get_solarposition

# We must normalize in such a way that there are many zeros 

# Paths
y_path = "data/Raw/pv_dataset_full.xlsx"
x_path = "data/Raw/wx_dataset_full.xlsx"
OUT_DIR = Path("../data/Processed")

def normalize_dt_df(df, col=None):
    df = df.copy()

    if col is None:
        df.index = pd.to_datetime(df.index, errors="coerce").floor("min")
        df = df[~df.index.isna()]
    else:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.floor("min")
        df = df.dropna(subset=[col]).set_index(col)

    return df


def load_excel(path: str, sheet: str | int = 0):
    return pd.read_excel(path, sheet_name=sheet)

def join_x_y(x_df, y_df, y_col="Energy"):
    """
    Joins X and Y into a single DataFrame by the temporal index.
    """
    assert x_df.index.equals(y_df.index), "X and Y are not aligned"

    df = x_df.copy()
    df[y_col] = y_df[y_col]
    return df


def save_joint_splits(train, val, test, out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # remove tz from index for Excel
    train = remove_index_timezone(train)
    val   = remove_index_timezone(val)
    test  = remove_index_timezone(test)

    train.to_excel(out_dir / "train.xlsx", index=True)
    val.to_excel(out_dir / "val.xlsx", index=True)
    test.to_excel(out_dir / "test.xlsx", index=True)

    print("Saved: train.xlsx, val.xlsx, test.xlsx")


def save_joint_splits_inference(x_df, y_df, out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # remove tz from index for Excel
    x_inference = remove_index_timezone(x_df)
    y_inference = remove_index_timezone(y_df)
    

    x_inference.to_excel(out_dir / "x_inference.xlsx", index=True)
    y_inference.to_excel(out_dir / "y_inference.xlsx", index=True)
    

    print("Saved: x_inference, y_inference")

def download_excel(url_or_path: str, sheet: str | int = 0) -> pd.DataFrame:
    """
    Downloads (if URL) or opens (if local path) an Excel and returns the specified sheet as a DataFrame.
    """
    return pd.read_excel(url_or_path, sheet_name=sheet)

def concatenate_sheets(df1: pd.DataFrame, df2: pd.DataFrame, axis: int = 0) -> pd.DataFrame:
    """
    Concatenates two DataFrames.
    axis=0 -> one below the other (same columns)
    axis=1 -> side by side (same rows / index)
    """
    return pd.concat([df1, df2], axis=axis, ignore_index=(axis == 0))


def fill_weather_description(x_df, value="unknown"):
    x_df = x_df.copy()
    x_df["weather_description"] = x_df["weather_description"].fillna(value)
    return x_df


def factorize_weather_description(x_df, col="weather_description", sort=True):
    x_df = x_df.copy()
    ids, vocab = pd.factorize(x_df[col], sort=sort)
    x_df[col] = ids
    return x_df

def transform_y(y):
    y = y.copy()
    y["Energy"] = np.log1p(y["Energy"])
    return y


def parse_dt_iso_16(x_df, col="dt_iso", target_tz=None):
    """
    - Keeps the first 16 chars (YYYY-MM-DD HH:MM)
    - Parses in UTC (tz-aware)
    - If target_tz is not None, converts to that tz
    """
    x_df = x_df.copy()

    s = (
        x_df[col].astype("string").str.strip().str[:16].str.replace("T", " ", regex=False)
    )

    s = pd.to_datetime(s, format="%Y-%m-%d %H:%M", errors="coerce")

    s = s.dt.tz_localize(target_tz)

    x_df[col] = s
    return x_df


def delete_invalid_dates(x_df, col):
    return x_df.dropna(subset=[col])


def extract_circular_features(df, dt_col="dt_iso"):
    """
    Extracts cyclic characteristics. 
    If dt_col is None or not in columns, uses the DataFrame index.
    """
    df = df.copy()

    # 1. Time source selection logic
    if dt_col is not None and dt_col in df.columns:
        # If column exists, we use it
        dt_series = df[dt_col].dt
    else:
        # If dt_col is None or moved to index, use index
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)
        dt_series = df.index

    # 2. Extract components
    hour = dt_series.hour
    month = dt_series.month
    weekday = dt_series.weekday

    # 3. Circular calculations
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    df["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)

    return df


def sort_and_index_by_date(df, col="dt_iso"):
    return df.sort_values(col).set_index(col)


def remove_index_timezone(df):
    """
    Removes tz from index without shifting hours (Excel-friendly).
    """
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def replace_na(df, col):
    df = df.copy()
    if col in df.columns:
        df[col] = df[col].fillna(0)
    return df


def delete_non_informative_columns(x_df, cols_to_delete):
    x_df = x_df.copy()
    existing_cols = [c for c in cols_to_delete if c in x_df.columns]
    if existing_cols:
        x_df.drop(columns=existing_cols, inplace=True)
    return x_df


# -----------------------------
# Split by indices (Sequential)
# -----------------------------
def split_by_indices(df, train_frac=0.7, val_frac=0.15):
    """
    SEQUENTIAL split:
      train = first train_frac
      val   = next val_frac
      test  = remainder
    """
    df = df.copy()
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end].copy()
    val   = df.iloc[train_end:val_end].copy()
    test  = df.iloc[val_end:].copy()

    return train, val, test

def add_solar_position(df, lat=40.4168, lon=-3.7038):
    """
    Calculates solar elevation and azimuth based on temporal index.
    """
    df = df.copy()

    if df.index.duplicated().any():
        print(f"⚠️ Detected {df.index.duplicated().sum()} duplicate rows. Removing...")
        df = df[~df.index.duplicated(keep='first')]

    # Ensure index is DatetimeIndex and has TZ
    if df.index.tz is None:
        times = df.index.tz_localize('UTC')
    else:
        times = df.index

    # Get solar position
    solpos = get_solarposition(times, lat, lon)
    
    df["solar_elevation"] = solpos["apparent_elevation"]
    df["solar_azimuth"] = solpos["azimuth"]
    
    # Circular features for azimuth
    df["solar_azimuth_sin"] = np.sin(np.radians(solpos["azimuth"]))
    df["solar_azimuth_cos"] = np.cos(np.radians(solpos["azimuth"]))
    
    return df


# -----------------------------
# Normalization WITHOUT leakage
# -----------------------------
def fit_stats(train_df):
    # Continuous columns (NOT binary nor cyclic)
    numeric_cols = []
    for c in train_df.columns:
        # do not normalize binary
        if train_df[c].dropna().isin([0, 1]).all():
            continue

        # do not normalize sin/cos
        if c.endswith("_sin") or c.endswith("_cos"):
            continue

        numeric_cols.append(c)

    mu = train_df[numeric_cols].mean()
    std = train_df[numeric_cols].std().replace(0, 1.0)
    return numeric_cols, mu, std


def apply_stats(df: pd.DataFrame, numeric_cols, mu, std):
    df = df.copy()
    df[numeric_cols] = (df[numeric_cols] - mu) / std
    return df

def load_stats(out_dir=OUT_DIR, filename="stats.pkl"):
    with open(Path(out_dir) / filename, "rb") as f:
        stats = pickle.load(f)

    return (
        stats["X"]["numeric_cols"],
        stats["X"]["mu"],
        stats["X"]["std"],
        stats["Y"]["numeric_cols"],
        stats["Y"]["mu"],
        stats["Y"]["std"],
    )


# -----------------------------
# X and Y Preparation
# -----------------------------
def prepare_x_df(x_df, target_tz=None, null_weather_val="unknown"):
    x_df = x_df.copy()

    real_lat = x_df['lat'].iloc[0]
    real_lon = x_df['lon'].iloc[0]
    print(f"Using latitude: {real_lat}, longitude: {real_lon} for solar calculation.")

    x_df = fill_weather_description(x_df, value=null_weather_val)
    x_df = factorize_weather_description(x_df, col="weather_description", sort=True)
    
    # parse datetime
    x_df = parse_dt_iso_16(x_df, target_tz=target_tz)
    x_df = delete_invalid_dates(x_df, col="dt_iso")
    x_df = sort_and_index_by_date(x_df, col="dt_iso")

    if x_df.index.duplicated().any():
        print(f"⚠️ Detected {x_df.index.duplicated().sum()} duplicate rows. Removing...")
        x_df = x_df[~x_df.index.duplicated(keep='first')]
    
    # Solar position
    x_df = add_solar_position(x_df, lat=real_lat, lon=real_lon)
    
    # cyclic features
    x_df = extract_circular_features(x_df, dt_col=None)
    
    # delete non informative
    x_df = delete_non_informative_columns(x_df, cols_to_delete=["lat", "lon"])

    # NA
    x_df = replace_na(x_df, "rain_1h")
    
    return x_df


def split_data(x_df, train_frac=0.8, val_frac=0.10):
    train, val, test = split_by_indices(x_df, train_frac=train_frac, val_frac=val_frac)
    return train, val, test


def normalize_df(train, val, test, remove_tz_excel=True):
    # normalize without leakage
    numeric_cols, mu, std = fit_stats(train)
    train = apply_stats(train, numeric_cols, mu, std)
    val   = apply_stats(val, numeric_cols, mu, std)
    test  = apply_stats(test, numeric_cols, mu, std)

    if remove_tz_excel:
        train = remove_index_timezone(train)
        val   = remove_index_timezone(val)
        test  = remove_index_timezone(test)
    return train, val, test 

def split_x_and_normalize_df(x_df, train_frac=0.7, val_frac=0.15, remove_tz_excel=True):
    train, val, test = split_by_indices(x_df, train_frac=train_frac, val_frac=val_frac)

    numeric_cols, mu, std = fit_stats(train)
    train = apply_stats(train, numeric_cols, mu, std)
    val   = apply_stats(val, numeric_cols, mu, std)
    test  = apply_stats(test, numeric_cols, mu, std)

    if remove_tz_excel:
        train = remove_index_timezone(train)
        val   = remove_index_timezone(val)
        test  = remove_index_timezone(test)
    return train, val, test


def normalize_single_df(x_df, numeric_cols, mu, std, remove_tz_excel=True):
    x_df = apply_stats(x_df, numeric_cols, mu, std)
    if remove_tz_excel:
        x_df = remove_index_timezone(x_df)
    return x_df

def prepare_y_df(y_df):
    """
    Rename columns and prepare index.
    """
    y_df = y_df.copy()
    y_df = y_df.rename(columns={'Max kWp':'dt_iso', 82.41:'Energy'})

    # date parse
    y_df["dt_iso"] = pd.to_datetime(y_df["dt_iso"], errors="coerce")
    y_df = y_df.dropna(subset=["dt_iso"])

    # sort and index
    y_df = sort_and_index_by_date(y_df, col="dt_iso")
    return y_df


def normalize_hourly_index(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index).dt.floor("H")
    return df


def split_y(y_df, train_frac=0.8, val_frac=0.10):
    train, val, test = split_by_indices(y_df, train_frac=train_frac, val_frac=val_frac)
    return train, val, test


def split_y_and_normalize_df(y_df, train_frac=0.7, val_frac=0.15):
    train, val, test = split_by_indices(y_df, train_frac=train_frac, val_frac=val_frac)

    numeric_cols, mu, std = fit_stats(train)
    train = apply_stats(train, numeric_cols, mu, std)
    val   = apply_stats(val, numeric_cols, mu, std)
    test  = apply_stats(test, numeric_cols, mu, std)

    train = remove_index_timezone(train)
    val   = remove_index_timezone(val)
    test  = remove_index_timezone(test)

    return train, val, test

def pipeline():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # load raw
    x_df_0 = load_excel(x_path, sheet=0)
    y_df_0 = load_excel(y_path, sheet=0)
    x_df_1 = load_excel(x_path, sheet=1)
    y_df_1 = load_excel(y_path, sheet=1)

    x_df = pd.concat([x_df_0, x_df_1], axis=0)
    y_df = pd.concat([y_df_0, y_df_1], axis=0)

    # prepare separately
    x_df = prepare_x_df(x_df)
    y_df = prepare_y_df(y_df)

    # ==================================================
    # ============ MASE SCALE (REAL SERIES) ==============
    # ==================================================
    y_energy_real = y_df["Energy"].values

    m = 24  # daily seasonality (hourly)
    naive_diff = np.abs(y_energy_real[m:] - y_energy_real[:-m])
    mase_scale = np.mean(naive_diff)

    print(f"MASE scale (real series): {mase_scale:.4f}")

    # Align by time
    x_df, y_df = x_df.align(y_df, join="inner", axis=0)

    # split X
    x_train, x_val, x_test = split_data(x_df, train_frac=0.8, val_frac=0.10)
    x_cols, x_mu, x_std = fit_stats(x_train)
    x_train = normalize_single_df(x_train, x_cols, x_mu, x_std, remove_tz_excel=True)
    x_val = normalize_single_df(x_val, x_cols, x_mu, x_std, remove_tz_excel=True)
    x_test = normalize_single_df(x_test, x_cols, x_mu, x_std, remove_tz_excel=True)
    
    # split Y
    y_train, y_val, y_test = split_data(y_df, train_frac=0.8, val_frac=0.10)
    y_cols, y_mu, y_std = fit_stats(y_train)
    y_train = normalize_single_df(y_train, y_cols, y_mu, y_std, remove_tz_excel=True)
    y_val = normalize_single_df(y_val, y_cols, y_mu, y_std, remove_tz_excel=True)
    y_test = normalize_single_df(y_test, y_cols, y_mu, y_std, remove_tz_excel=True)

    # Join X and Y
    train = join_x_y(x_train, y_train)
    val   = join_x_y(x_val, y_val)
    test  = join_x_y(x_test, y_test)

    # save sets
    save_joint_splits(train, val, test, OUT_DIR)
    print("Train:", train.shape)
    print("Val:", val.shape)
    print("Test:", test.shape)
    
    print("\n--- Starting Inference Normalization ---")
    
    # 1. Load raw data for inference
    x_df_2 = load_excel(x_path, sheet=2)
    y_df_2 = load_excel(y_path, sheet=2)

    # 2. Preprocessing
    x_df_inf = prepare_x_df(x_df_2)
    y_df_inf = prepare_y_df(y_df_2)

    # 3. Temporal alignment
    x_df_inf = normalize_dt_df(x_df_inf)
    y_df_inf = normalize_dt_df(y_df_inf)
    x_df_inf, y_df_inf = x_df_inf.align(y_df_inf, join="inner", axis=0)

    # 5. Normalize X
    x_df_inf = apply_stats(x_df_inf, x_cols, x_mu, x_std)
    
    # 6. Normalize Y
    y_df_inf = apply_stats(y_df_inf, y_cols, y_mu, y_std)

    # 7. Remove Timezone
    x_df_inf = remove_index_timezone(x_df_inf)
    y_df_inf = remove_index_timezone(y_df_inf)

    # 8. Join and Save
    inference = join_x_y(x_df_inf, y_df_inf)
    inference.to_excel(OUT_DIR / "inference.xlsx", index=True)

    # ==================================================
    # ============ SAVE STATS FOR INFERENCE =========
    # ==================================================

    stats = {
        "X": {
            "numeric_cols": x_cols,
            "mu": x_mu,
            "std": x_std,
        },
        "Y": {
            "numeric_cols": y_cols,
            "mu": y_mu,
            "std": y_std,
        },
        "mase": {
            "scale": mase_scale,
            "m": m,
        },
    }

    with open(OUT_DIR / "stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    print("stats.pkl saved successfully in data/Processed/")
    print("X and Y variables normalized successfully.")
    print("Processed columns in X:", x_cols)
    print("Example values in X (should be near 0):")
    print(x_df_inf[x_cols].head(3))


def main():
    pipeline()

if __name__ == "__main__":
    main()
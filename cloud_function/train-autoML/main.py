# train-autoML/main.py
import os, io, json, logging, traceback, time
import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from tpot import TPOTRegressor
import joblib
import matplotlib.pyplot as plt

# ---- ENV ----
PROJECT_ID     = os.getenv("PROJECT_ID", "")
GCS_BUCKET     = os.getenv("GCS_BUCKET", "")
DATA_KEY       = os.getenv("DATA_KEY", "structured/datasets/listings_master.csv")
OUTPUT_PREFIX  = os.getenv("OUTPUT_PREFIX", "preds")
TIMEZONE       = os.getenv("TIMEZONE", "America/New_York")
LOG_LEVEL      = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")

# ---- Helpers ----
def _read_csv_from_gcs(client, bucket, key):
    blob = client.bucket(bucket).blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket}/{key} not found")
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))

def _write_csv_to_gcs(client, bucket, key, df):
    blob = client.bucket(bucket).blob(key)
    blob.upload_from_string(df.to_csv(index=False), content_type="text/csv")

def _clean_numeric(s):
    s = s.astype(str).str.replace(r"[^\d.]+", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")

# ---- Main ----
def run_once(dry_run=False):
    client = storage.Client(project=PROJECT_ID)
    df = _read_csv_from_gcs(client, GCS_BUCKET, DATA_KEY)

    required = {
        "scraped_at", "price", "make", "model", "year", "mileage",
        "cylinders", "color", "condition", "transmission", "fuel", "title_status"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # --- Time processing ---
    dt = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
    df["scraped_at_local"] = dt.dt.tz_convert(TIMEZONE)
    df["date_local"] = df["scraped_at_local"].dt.date

    # --- Clean numerics ---
    df["price_num"]     = _clean_numeric(df["price"])
    df["year_num"]      = _clean_numeric(df["year"])
    df["mileage_num"]   = _clean_numeric(df["mileage"])
    df["cylinders_num"] = _clean_numeric(df["cylinders"])

    # --- Train / Holdout split ---
    unique_dates = sorted(df["date_local"].dropna().unique())
    if len(unique_dates) < 2:
        return {"status": "noop", "reason": "need at least two dates"}

    today_local = unique_dates[-1]
    train_df   = df[df["date_local"] < today_local].copy()
    holdout_df = df[df["date_local"] == today_local].copy()
    train_df = train_df[train_df["price_num"].notna()]

    if len(train_df) < 40:
        return {"status": "noop", "reason": "too few training rows"}

    # --- Features ---
    target = "price_num"
    cat_cols = ["make", "model", "color", "condition", "transmission", "fuel", "title_status"]
    num_cols = ["year_num", "mileage_num", "cylinders_num"]
    feats = cat_cols + num_cols

    X_train = train_df[feats]
    y_train = train_df[target]

    # --- Preprocessing ---
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
    ])

    # --- autoML: TPOT Model ---
    tpot = TPOTRegressor(
        generations=3,
        population_size=15,
        verbosity=2,
        random_state=42,
        max_time_mins=10
    )

    pipe = Pipeline([
        ("pre", pre),
        ("model", tpot)
    ])

    start = time.time()
    pipe.fit(X_train, y_train)
    logging.info("Training completed in %.2f seconds", time.time() - start)

    # --- Predictions ---
    preds_df = pd.DataFrame()
    mae_today = None

    if not holdout_df.empty:
        X_h = holdout_df[feats]
        y_true = holdout_df["price_num"]
        y_hat = pipe.predict(X_h)

        preds_df = holdout_df[[
            "post_id", "scraped_at", "make", "model", "year", "mileage", "price"
        ]].copy()
        preds_df["actual_price"] = y_true
        preds_df["pred_price"]   = np.round(y_hat, 2)

        mask = y_true.notna()
        if mask.any():
            mae_today = float(mean_absolute_error(y_true[mask], y_hat[mask]))

    # --- Save model ---
    model_path = "/tmp/tpot_pipeline.joblib"
    joblib.dump(pipe, model_path)

    # --- Permutation Importance ---
    perm_result = permutation_importance(pipe, X_train, y_train, n_repeats=10, random_state=42)
    perm_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance_mean": perm_result.importances_mean,
        "importance_std": perm_result.importances_std
    }).sort_values("importance_mean", ascending=False)
    perm_csv_path = "/tmp/permutation_importance.csv"
    perm_df.to_csv(perm_csv_path, index=False)
    client.bucket(GCS_BUCKET).blob(f"{OUTPUT_PREFIX}/permutation_importance.csv").upload_from_filename(perm_csv_path)

    # --- PDPs for top 3 features ---
    top_features = perm_df.head(3)["feature"].tolist()
    for f in top_features:
        fig, ax = plt.subplots(figsize=(6,4))
        PartialDependenceDisplay.from_estimator(pipe, X_train, features=[f], ax=ax)
        plt.tight_layout()
        png_path = f"/tmp/pdp_{f}.png"
        fig.savefig(png_path)
        plt.close(fig)
        client.bucket(GCS_BUCKET).blob(f"{OUTPUT_PREFIX}/pdp_{f}.png").upload_from_filename(png_path)

    # --- Save predictions ---
    now_utc = pd.Timestamp.utcnow().tz_convert("UTC")
    out_key = f"{OUTPUT_PREFIX}/{now_utc.strftime('%Y%m%d%H')}/preds.csv"
    if not dry_run and len(preds_df) > 0:
        _write_csv_to_gcs(client, GCS_BUCKET, out_key, preds_df)

    return {
        "status": "ok",
        "today_local": str(today_local),
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "mae_today": mae_today,
        "output_key": out_key,
        "permutation_importance": f"{OUTPUT_PREFIX}/permutation_importance.csv",
        "pdp_files": [f"{OUTPUT_PREFIX}/pdp_{f}.png" for f in top_features]
    }

# ---- HTTP Entrypoint ----
def train_autoML_http(request):
    try:
        body = request.get_json(silent=True) or {}
        result = run_once(dry_run=bool(body.get("dry_run", False)))
        return (json.dumps(result), 200, {"Content-Type": "application/json"})
    except Exception as e:
        logging.error(traceback.format_exc())
        return (json.dumps({"status": "error", "error": str(e)}), 500)

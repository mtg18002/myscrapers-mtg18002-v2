# train-autoML/main.py

import os, io, json, logging, time, traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from google.cloud import storage
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from tpot import TPOTRegressor
import joblib

# ---- ENV ----
PROJECT_ID     = os.getenv("PROJECT_ID", "")
GCS_BUCKET     = os.getenv("GCS_BUCKET", "")
DATA_KEY       = os.getenv("DATA_KEY", "structured/datasets/listings_master_llm.csv")
OUTPUT_PREFIX  = os.getenv("OUTPUT_PREFIX", "preds-autoML")  # Base path in GCS
TIMEZONE       = os.getenv("TIMEZONE", "America/New_York")
LOG_LEVEL      = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")

# ---- HELPERS ----
def _read_csv_from_gcs(client, bucket, key):
    blob = client.bucket(bucket).blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket}/{key} not found")
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))

def _write_csv_to_gcs(client, bucket, key, df):
    blob = client.bucket(bucket).blob(key)
    blob.upload_from_string(df.to_csv(index=False, index_label=False), content_type="text/csv")

def _clean_numeric(s):
    s = s.astype(str).str.replace(r"[^\d.]+", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")

def _append_csv_to_gcs(client, bucket, key, df_new):
    blob = client.bucket(bucket).blob(key)

    if blob.exists():
        existing_df = pd.read_csv(io.BytesIO(blob.download_as_bytes()))
        combined_df = pd.concat([existing_df, df_new], ignore_index=True)
    else:
        combined_df = df_new

    blob.upload_from_string(combined_df.to_csv(index=False), content_type="text/csv")

# ---- MAIN FUNCTION ----
def run_once(dry_run=False):
    client = storage.Client(project=PROJECT_ID)
    df = _read_csv_from_gcs(client, GCS_BUCKET, DATA_KEY)

    # Required columns
    required = {"scraped_at", "price", "make", "model", "year", "mileage",
                "cylinders", "color", "condition", "transmission", "fuel", "title_status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # --- Time processing ---
    dt = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
    df["scraped_at_local"] = dt.dt.tz_convert(TIMEZONE)
    df["date_local"] = df["scraped_at_local"].dt.date

    # --- Clean numeric features ---
    df["price_num"]     = _clean_numeric(df["price"])
    df["year_num"]      = _clean_numeric(df["year"])
    df["mileage_num"]   = _clean_numeric(df["mileage"])
    df["cylinders_num"] = _clean_numeric(df["cylinders"])

    # --- Train / Holdout split (all data < today) ---
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

    # --- TPOT autoML ---
    tpot = TPOTRegressor(generations=3, population_size=15, random_state=42, max_time_mins=10)
    pipe = Pipeline([("pre", pre), ("model", tpot)])

    start = time.time()
    pipe.fit(X_train, y_train)
    logging.info("Training completed in %.2f seconds", time.time() - start)

    # --- Holdout predictions ---
    preds_df = pd.DataFrame()
    mae_today = None
    rmse_today = None
    mape_today = None
    bias_today = None
    if not holdout_df.empty:
        X_h = holdout_df[feats]
        y_true = holdout_df["price_num"]
        y_hat = pipe.predict(X_h)

        preds_df = holdout_df[["post_id", "scraped_at", "make", "model", "year", "mileage", "price"]].copy()
        preds_df["actual_price"] = y_true
        preds_df["pred_price"]   = np.round(y_hat, 2)

        mask = y_true.notna()
        if mask.any():
            mae_today = float(mean_absolute_error(y_true[mask], y_hat[mask]))
            rmse_today = float(root_mean_squared_error(y_true[mask], y_hat[mask]))
            mape_today = float(np.mean(np.abs((y_true[mask] - y_hat[mask]) / y_true[mask])) * 100)
            bias_today = float(np.mean(y_hat[mask] - y_true[mask]))

    # --- Timestamp for saving outputs ---
    timestamp = pd.Timestamp.utcnow().strftime("%Y%m%d%H%M%S")
    base_path = f"/tmp/{timestamp}"
    os.makedirs(base_path, exist_ok=True)

    metrics_df = pd.DataFrame([{
        "timestamp": timestamp,
        "mae": mae_today,
        "rmse": rmse_today,
        "mape": mape_today,
        "bias": bias_today,
        "n_train": len(train_df),
        "n_holdout": len(holdout_df)
     }])

    # --- Save predictions ---
    preds_path = f"{base_path}/preds_{timestamp}.csv"
    preds_df.to_csv(preds_path, index=False)

    # --- Permutation Importance (ALL features) ---
    X_h_transformed = pipe.named_steps['pre'].transform(X_h)
    feature_names = pipe.named_steps['pre'].get_feature_names_out()
    X_h_transformed_df = pd.DataFrame(X_h_transformed, columns=feature_names)
    best_model = pipe.named_steps['model'].fitted_pipeline_
    result = permutation_importance(best_model, X_h_transformed, y_true, n_repeats=20, random_state=42, scoring="neg_mean_absolute_error")

    perm_df = pd.DataFrame({
        "feature": feature_names,
        "importances_mean": result.importances_mean
    })
    perm_csv_path = f"{base_path}/perm_importance_{timestamp}.csv"
    perm_df.to_csv(perm_csv_path, index=False)

    # Boxplot
    perm_idx_sorted = np.argsort(result.importances_mean)
    fig, ax = plt.subplots(figsize=(6, max(4, len(feats)*0.3)))
    ax.boxplot(result.importances.T, vert=False, labels=feature_names)
    fig.suptitle("Permutation Importance (All Features)", y=1.05)
    fig.tight_layout()
    perm_plot_path = f"{base_path}/perm_importance_{timestamp}.png"
    fig.savefig(perm_plot_path)
    plt.close(fig)

    # --- PDPs for top 3 features ---
    top3_idx = np.argsort(result.importances_mean)[-3:]
    top3_features = X_h_transformed_df.columns[top3_idx]
    pdp_paths = []
    for feat in top3_features:
        fig, ax = plt.subplots(figsize=(6, 4))
        display = PartialDependenceDisplay.from_estimator(best_model, X_h_transformed_df, [feat], ax=ax)
        fig.tight_layout()
        pdp_path = f"{base_path}/pdp_{feat}_{timestamp}.png"
        fig.savefig(pdp_path)
        plt.close(fig)
        pdp_paths.append(pdp_path)

    # --- Save pipeline ---
    model_path = f"{base_path}/tpot_pipeline_{timestamp}.joblib"
    joblib.dump(pipe, model_path)

    # --- upload to GCS ---
    if not dry_run:
        gcs_base = f"{OUTPUT_PREFIX}/{timestamp}"
        metrics_global_path = f"{OUTPUT_PREFIX}/metrics/model_accuracy.csv" 
        _write_csv_to_gcs(client, GCS_BUCKET, f"{gcs_base}/preds_df.csv", preds_df)
        _write_csv_to_gcs(client, GCS_BUCKET, f"{gcs_base}/perm_importance.csv", perm_df)
        perm_plot_blob = client.bucket(GCS_BUCKET).blob(f"{gcs_base}/perm_importance.png")
        perm_plot_blob.upload_from_filename(perm_plot_path)
        for pdp_path in pdp_paths:
            blob_name = f"{gcs_base}/{os.path.basename(pdp_path)}"
            client.bucket(GCS_BUCKET).blob(blob_name).upload_from_filename(pdp_path)
        pipeline_blob = client.bucket(GCS_BUCKET).blob(f"{gcs_base}/tpot_pipeline_{timestamp}.joblib")
        pipeline_blob.upload_from_filename(model_path)
        _append_csv_to_gcs(client, GCS_BUCKET, metrics_global_path, metrics_df)

    return {
        "status": "ok",
        "timestamp": timestamp,
        "mae_today": mae_today,
        "preds_path": preds_path,
        "perm_csv_path": perm_csv_path,
        "perm_plot_path": perm_plot_path,
        "pdp_paths": pdp_paths,
        "model_path": model_path
    }

# ---- HTTP Entrypoint ----
def train_autoML_http(request):
    try:
        body = request.get_json(silent=True) or {}
        result = run_once(dry_run=bool(body.get("dry_run", False)))
        return (json.dumps(result), 200, {"Content-Type": "application/json"})
    except Exception as e:
        logging.error(traceback.format_exc())
        return (json.dumps({"status": "error", "error": str(e)}), 500, {"Content-Type": "application/json"})

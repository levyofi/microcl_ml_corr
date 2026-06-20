import os
import pandas as pd
import numpy as np
import sys

PROJECT_ROOT = "/home/ofir/Dropbox/Antigravity/NichMapR_ml_corr"
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

import microclimate_corr.preprocessing as pr
from microclimate_corr.schema import CONTINUOS_MICROHABITAT

def extract_beach_splits():
    data_path = os.path.join(PROJECT_ROOT, "data/experiments_data/Beach_data_preprocessed.csv")
    print(f"Loading Beach data from {data_path}...")
    data = pr.load_prepared_csv_data(data_path, CONTINUOS_MICROHABITAT, "%Y-%m-%d %H:%M:%S", includes_index=True)
    
    # Run the exact split logic
    train_df, val_df, test_df = pr.stratified_train_val_test_split(
        data,
        training_perc_split=0.75,
        val_perc_split=0.125,
        stratify_col="location",
        n_days_split=7,
        datetime_col="time",
        seed=123
    )
    
    # Map back to assign splits
    train_times = pd.DataFrame({"time": train_df["time"], "time_series_site": train_df["time_series_site"], "split": "train"})
    val_times = pd.DataFrame({"time": val_df["time"], "time_series_site": val_df["time_series_site"], "split": "val"})
    test_times = pd.DataFrame({"time": test_df["time"], "time_series_site": test_df["time_series_site"], "split": "test"})
    
    combined = pd.concat([train_times, val_times, test_times])
    out_path = os.path.join(PROJECT_ROOT, "data/experiments_data/beach_splits.csv")
    combined.to_csv(out_path, index=False)
    print(f"Saved Beach splits ({len(combined)} rows) to {out_path}")

def extract_desert_splits():
    data_path = os.path.join(PROJECT_ROOT, "data/experiments_data/desert_data_preprocessed.csv")
    print(f"Loading Desert data from {data_path}...")
    data = pr.load_prepared_csv_data(data_path, CONTINUOS_MICROHABITAT, "%Y-%m-%d %H:%M:%S", includes_index=True)
    data['Location'] = data['Location'].replace({'Mishamr': 'Mishmar', 'Mishmar-': 'Mishmar'})
    data = data.rename(columns={'Location': 'location', 'Season': 'season', 'Object': 'object', 'Size': 'size'})
    
    # Run the exact split logic
    train_df, val_df, test_df = pr.stratified_train_val_test_split(
        data,
        training_perc_split=0.75,
        val_perc_split=0.125,
        stratify_col="location",
        n_days_split=7,
        datetime_col="time",
        seed=123
    )
    
    # Map back to assign splits
    train_times = pd.DataFrame({"time": train_df["time"], "site_id": train_df["site_id"], "split": "train"})
    val_times = pd.DataFrame({"time": val_df["time"], "site_id": val_df["site_id"], "split": "val"})
    test_times = pd.DataFrame({"time": test_df["time"], "site_id": test_df["site_id"], "split": "test"})
    
    combined = pd.concat([train_times, val_times, test_times])
    out_path = os.path.join(PROJECT_ROOT, "data/experiments_data/desert_splits.csv")
    combined.to_csv(out_path, index=False)
    print(f"Saved Desert splits ({len(combined)} rows) to {out_path}")

if __name__ == "__main__":
    extract_beach_splits()
    extract_desert_splits()

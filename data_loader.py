import pandas as pd

def load_tabular_data(path="data/medical_data.csv"):
    data = pd.read_csv(path)
    X = data.drop(columns=["disease_label"])
    y = data["disease_label"]
    return X, y

def load_time_series_data(path="data/medical_time_series.csv"):
    df = pd.read_csv(path)
    df["time"] = df["time"].astype(int)
    df["patient_id"] = df["patient_id"].astype(str)
    return df

import os
import pandas as pd
from scipy.io import arff


def load_data(data_name):
    file_path = f"datasets/{data_name}.arff"
    if os.path.exists(file_path):
        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data).apply(lambda x: pd.to_numeric(x, errors="ignore"))
        X = pd.get_dummies(df.loc[:, df.columns != "Class"]).values
        unique_labels = df["Class"].unique()
        labels_dict = dict(zip(unique_labels, range(len(unique_labels))))
        df.loc[:, "Class"] = df.applymap(lambda s: labels_dict.get(s) if s in labels_dict else s)
        y = df["Class"].values
        return X, y
    return [], []

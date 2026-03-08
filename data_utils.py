import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
BASE_DIR = r"E:\HAC"

def load_and_partition_data(client_id: int, total_clients: int = 3):
    """
    Load RAW data, partition it among clients,
    and apply LOCAL scaling (federated-style).
    Returns ONLY 3 values.
    """

    data_path = os.path.join(
        BASE_DIR,
        "data",
        "heart_attack_fresh.csv"
    )

    df = pd.read_csv(data_path)

    X = df.drop("Heart_disease", axis=1).values
    y = df["Heart_disease"].values

    input_dim = X.shape[1]

    # Reproducible shuffle
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Client-wise split
    part_size = len(X) // total_clients
    start = (client_id - 1) * part_size
    end = len(X) if client_id == total_clients else client_id * part_size

    X_local = X[start:end]
    y_local = y[start:end]

    # Local scaling
    scaler = StandardScaler()
    X_local_scaled = scaler.fit_transform(X_local)

    return X_local_scaled, y_local, input_dim

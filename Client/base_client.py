import sys
import os

BASE_DIR = r"E:\HAC"
sys.path.insert(0, BASE_DIR)

import flwr as fl
import tensorflow as tf

from data_utils import load_and_partition_data
from hybrid_model import HybridModel


class HeartClient(fl.client.NumPyClient):

    def __init__(self, X_local, y_local):
        self.X_local = X_local
        self.y_local = y_local
        self.model = HybridModel(input_dim=X_local.shape[1])

    def get_parameters(self, config):
        return self.model.get_dnn_weights()

    def fit(self, parameters, config):
        # Set global weights
        self.model.set_dnn_weights(parameters)

        # Local training
        self.model.fit(self.X_local, self.y_local, epochs=7)

        return self.model.get_dnn_weights(), len(self.X_local), {}


def run(client_id: int):
    print(f"--- Client {client_id}: Loading data partition... ---")

    X_local, y_local, input_dim = load_and_partition_data(client_id)

    print(
        f"--- Client {client_id}: Data loaded "
        f"({len(X_local)} samples). Starting client... ---"
    )

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8081",
        client=HeartClient(X_local, y_local),
    )

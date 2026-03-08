# import sys
# import os

# BASE_DIR = r"E:\HAC"
# sys.path.append(BASE_DIR)

# import flwr as fl
# import tensorflow as tf
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from hybrid_model import create_dnn_model


# DATA_PATH = os.path.join(BASE_DIR, "data", "heart_attack_fresh.csv")

# FL_ROUNDS = 20
# NUM_CLIENTS = 3
# RANDOM_SEED = 42

# # SAVED_MODEL_PATH = os.path.join(
# #     BASE_DIR, "model", "global_dnn_model.keras"
# # )

# SAVED_MODEL_PATH = os.path.join(
#     BASE_DIR, "model", "global_dnn_model.h5"
# )


# BEST_ACCURACY = 0.0

# # ---------------- SERVER EVALUATION FN ----------------
# def get_server_evaluate_fn(X_test, y_test, input_dim):

#     def evaluate(server_round, parameters, config):
#         global BEST_ACCURACY

#         model = create_dnn_model(input_dim)
#         model.set_weights(parameters)

#         loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

#         print(
#             f"[Server] Round {server_round} | "
#             f"Global DNN Accuracy: {accuracy:.4f}"
#         )

#         # ✅ SAVE BEST MODEL (EASY LOGIC)
#         if accuracy > BEST_ACCURACY:
#             BEST_ACCURACY = accuracy
#             os.makedirs(os.path.dirname(SAVED_MODEL_PATH), exist_ok=True)
#             model.save(SAVED_MODEL_PATH)
#             print(
#                 f"🏆 Best model updated "
#                 f"(Accuracy = {accuracy:.4f})"
#             )

#         return loss, {"accuracy": accuracy}

#     return evaluate


# def main():
#     print("🚀 Starting Federated Learning Server")

#     df = pd.read_csv(DATA_PATH)

#     X = df.drop("Heart_disease", axis=1).values
#     y = df["Heart_disease"].values
#     input_dim = X.shape[1]

#     np.random.seed(RANDOM_SEED)
#     indices = np.random.permutation(len(X))

#     split = int(0.8 * len(X))
#     test_idx = indices[split:]

#     X_test = X[test_idx]
#     y_test = y[test_idx]

#     # -------- Server-side scaling (ONLY for evaluation) --------
#     scaler = StandardScaler()
#     X_test_scaled = scaler.fit_transform(X_test)

#     # -------- Initialize global model --------
#     temp_model = create_dnn_model(input_dim)
#     initial_parameters = fl.common.ndarrays_to_parameters(
#         temp_model.get_weights()
#     )
#     del temp_model

#     # -------- Federated Strategy --------
#     strategy = fl.server.strategy.FedAvg(
#         initial_parameters=initial_parameters,
#         min_fit_clients=NUM_CLIENTS,
#         min_available_clients=NUM_CLIENTS,
#         evaluate_fn=get_server_evaluate_fn(
#             X_test_scaled, y_test, input_dim
#         ),
#     )

#     # -------- Start Server --------
#     fl.server.start_server(
#         server_address="0.0.0.0:8081",
#         config=fl.server.ServerConfig(num_rounds=FL_ROUNDS),
#         strategy=strategy,
#     )


# if __name__ == "__main__":
#     os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#     tf.get_logger().setLevel("ERROR")
#     main()
import sys
import os

BASE_DIR = r"E:\HAC"
sys.path.append(BASE_DIR)

import flwr as fl
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from hybrid_model import create_dnn_model

# ---------------- CONFIGURATION ----------------
DATA_PATH = os.path.join(BASE_DIR, "data", "heart_attack_fresh.csv")
FL_ROUNDS = 20
NUM_CLIENTS = 3
RANDOM_SEED = 42

SAVED_MODEL_PATH = os.path.join(
    BASE_DIR, "model", "global_dnn_model.h5"
)


# SAVED_WEIGHTS_NPY = os.path.join(BASE_DIR, "model", "best_weights.npy")

BEST_ACCURACY = 0.0

# ---------------- SERVER EVALUATION FUNCTION ----------------
def get_server_evaluate_fn(X_test, y_test, input_dim):

    def evaluate(server_round, parameters, config):
        global BEST_ACCURACY

        model = create_dnn_model(input_dim)
        model.set_weights(parameters)

        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

        print(
            f"[Server] Round {server_round} | "
            f"Global DNN Accuracy: {accuracy:.4f}"
        )

        if accuracy > BEST_ACCURACY:
            BEST_ACCURACY = accuracy
            os.makedirs(os.path.dirname(SAVED_MODEL_PATH), exist_ok=True)
            
            try:
                model.save(SAVED_MODEL_PATH, save_format='h5')
                print(f"Best Model saved: {SAVED_MODEL_PATH} (Acc: {accuracy:.4f})")
            except Exception as e:
                print(f"Error saving model: {e}")
        return loss, {"accuracy": accuracy}

    return evaluate


# ---------------- MAIN FUNCTION ----------------
def main():
    print("Starting Federated Learning Server...")

    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)

    X = df.drop("Heart_disease", axis=1).values
    y = df["Heart_disease"].values
    input_dim = X.shape[1]

    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(len(X))
    
    split = int(0.8 * len(X))
    test_idx = indices[split:]
    
    X_test = X[test_idx]
    y_test = y[test_idx]

    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)
    
    # Global Model Init
    temp_model = create_dnn_model(input_dim)
    initial_parameters = fl.common.ndarrays_to_parameters(
        temp_model.get_weights()
    )
    del temp_model

    strategy = fl.server.strategy.FedAvg(
        initial_parameters=initial_parameters,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=get_server_evaluate_fn(
            X_test_scaled, y_test, input_dim
        ),
    )

    fl.server.start_server(
        server_address="0.0.0.0:8081",
        config=fl.server.ServerConfig(num_rounds=FL_ROUNDS),
        strategy=strategy,
    )

if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.get_logger().setLevel("ERROR")
    main()
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit_machine_learning.algorithms import VQC
from qiskit.utils import QuantumInstance
from qiskit.algorithms.optimizers import COBYLA
from qiskit import Aer
import time
import numpy as np

def train_quantum(X_train, X_test, y_train, y_test, on_iteration=None):
    """
    Trains a Variational Quantum Classifier (VQC) and returns training metrics and artifacts.
    """
    objective_func_vals = []

    def callback_graph(weights, obj_func_eval):
        objective_func_vals.append(obj_func_eval)
        if on_iteration is not None:
            on_iteration(len(objective_func_vals))

    backend = Aer.get_backend("aer_simulator")
    qi = QuantumInstance(backend, shots=50, seed_simulator=42, seed_transpiler=42)

    # Preprocessing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=4)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Quantum circuit setup
    feature_map = ZZFeatureMap(feature_dimension=4, reps=2)
    ansatz = TwoLocal(4, ['ry', 'rz'], 'cz', reps=2)
    optimizer = COBYLA(maxiter=20)

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        quantum_instance=qi,
        callback=callback_graph
    )

    # Training
    start = time.time()
    vqc.fit(X_train_pca, y_train.to_numpy().ravel())
    elapsed = time.time() - start

    # Predictions
    y_pred = vqc.predict(X_test_pca)
    y_true = y_test.to_numpy().ravel()

    # Metrics
    metrics = {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1 Score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "Confusion Matrix": confusion_matrix(y_true, y_pred).tolist(),  # for JSON serializability
        "Classification Report": classification_report(y_true, y_pred, output_dict=True)
    }

    return metrics, elapsed, objective_func_vals, vqc.weights, scaler, pca

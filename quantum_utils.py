from joblib import load
import numpy as np
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit_machine_learning.algorithms import VQC
from qiskit.utils import QuantumInstance
from qiskit.algorithms.optimizers import COBYLA
from qiskit import Aer

def load_quantum_model(uploaded_model, X_test, y_test=None):
    """
    Loads a quantum model from saved weights and preprocessing,
    rebuilds the VQC, and evaluates it on X_test.

    Parameters:
    - uploaded_model: joblib file-like object containing 'weights', 'scaler', 'pca'
    - X_test: raw test features (DataFrame or array)
    - y_test: labels (optional) – if provided, returns accuracy

    Returns:
    - model: reconstructed VQC model
    - predictions: model.predict(X_test)
    - accuracy (optional): if y_test is provided
    """
    # Load components
    data = load(uploaded_model)
    weights = data['weights']
    scaler = data['scaler']
    pca = data['pca']

    # Preprocess
    X_test_scaled = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_scaled)

    # Rebuild VQC model
    feature_map = ZZFeatureMap(feature_dimension=4, reps=2)
    ansatz = TwoLocal(4, ['ry', 'rz'], 'cz', reps=2)
    optimizer = COBYLA(maxiter=1)  # No training
    qi = QuantumInstance(Aer.get_backend("aer_simulator"), shots=28, seed_simulator=42, seed_transpiler=42)

    model = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer, quantum_instance=qi)

    # Dummy fit to initialize internals
    model.fit(X_test_pca[:2], np.array([0, 1]))  # Fake labels
    model._ret['optimal_point'] = weights

    # Predict
    predictions = model.predict(X_test_pca)

    if y_test is not None:
        acc = model.score(X_test_pca, y_test.to_numpy().ravel())
        return model, predictions, acc

    return model, predictions

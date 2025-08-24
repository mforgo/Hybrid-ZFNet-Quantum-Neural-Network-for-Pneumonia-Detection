# models/hybrid_integration.py
# ---------------------------------------------------------------
# Working hybrid integration aligned with the reverted experiment:
# - Assumes quantum_experiment.py saved final_quantum_params.npy
# - Uses identical RX/RY/RZ ansatz (LAYERS x QUBITS x 3)
# - Loads 4-feature pipeline if available; otherwise falls back to
#   a synthetic test set so the script always runs end-to-end.
# ---------------------------------------------------------------

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

import pennylane as qml
from pennylane import numpy as pnp

# ------------------ CONFIG ------------------
SEED        = int(os.getenv("QSEED", "42"))
N_QUBITS    = 4
N_LAYERS    = int(os.getenv("LAYERS", "2"))  # must match experiment
RESULTS_DIR = "./results"
PARAMS_PATH = "final_quantum_params.npy"
FEATURES_CSV = "data/features/metadata.csv"  # optional; if missing we fallback to synthetic
TEST_SAMPLES = 300
SHOTS_EVAL   = os.getenv("EVAL_SHOTS")  # e.g., "256" to evaluate with shots; None => analytic

np.random.seed(SEED)

print(f"🔗 Hybrid integration | expect params from {PARAMS_PATH} | layers={N_LAYERS}")

# ------------------ LOAD PARAMS ------------------
if not os.path.exists(PARAMS_PATH):
    raise FileNotFoundError(f"{PARAMS_PATH} not found. Run models/quantum_experiment.py first.")

theta_flat = np.load(PARAMS_PATH)
expected = N_LAYERS * N_QUBITS * 3
if theta_flat.size != expected:
    raise ValueError(f"Loaded params have {theta_flat.size} entries; expected {expected} "
                     f"(LAYERS={N_LAYERS} QUBITS={N_QUBITS} ROT=3).")
theta = theta_flat.reshape(N_LAYERS, N_QUBITS, 3)
print(f"✓ Loaded parameters: shape={theta.shape}")

# ------------------ DATA PIPELINE ------------------
def load_features_if_available():
    """Try to use your existing 4-feature pipeline. If not available, return None."""
    if not os.path.exists(FEATURES_CSV):
        return None
    try:
        meta = pd.read_csv(FEATURES_CSV)
        X = np.vstack([np.load(fp) for fp in meta.feature_path])
        y = meta.label.values

        train_mask = meta.split == 'train'
        test_mask  = meta.split == 'test'
        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        selector = SelectKBest(score_func=f_classif, k=4)
        selector.fit(X_train, y_train)
        X_train_sel = selector.transform(X_train)
        X_test_sel  = selector.transform(X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)
        X_test_scaled  = scaler.transform(X_test_sel)

        # Map to [0, 2π] for quantum encoding
        def to_quantum(Xm):
            X_min, X_max = Xm.min(axis=0), Xm.max(axis=0)
            rng = X_max - X_min
            rng[rng == 0] = 1
            return (Xm - X_min) / rng * (2 * np.pi)

        X_train_q = to_quantum(X_train_scaled)
        X_test_q  = to_quantum(X_test_scaled)

        if len(X_test_q) > TEST_SAMPLES:
            idx = np.random.choice(len(X_test_q), TEST_SAMPLES, replace=False)
            X_test_q    = X_test_q[idx]
            X_test_scaled = X_test_scaled[idx]
            y_test      = y_test[idx]

        return X_train_scaled, y_train, X_test_scaled, y_test, X_train_q, X_test_q
    except Exception as e:
        print(f"⚠️ Feature pipeline present but failed to load cleanly: {e}")
        return None

def create_synthetic_testset(n_samples=400, n_features=4):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_clusters_per_class=2,
        weights=[0.6, 0.4],
        flip_y=0.01,
        class_sep=1.2,
        random_state=SEED
    )
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    # classical features identical to quantum pre-step (we test consistency)
    X_classical = Xs.copy()
    # quantum encoding range [0, 2π]
    X_min, X_max = Xs.min(axis=0), Xs.max(axis=0)
    rng = X_max - X_min
    rng[rng == 0] = 1
    X_quantum = (Xs - X_min) / rng * (2*np.pi)
    # train/test split
    Xc_tr, Xc_te, y_tr, y_te = train_test_split(X_classical, y, test_size=0.25, stratify=y, random_state=SEED)
    Xq_tr, Xq_te, _, _ = train_test_split(X_quantum, y, test_size=0.25, stratify=y, random_state=SEED)
    print("🧪 Using synthetic test set (features pipeline not found).")
    return Xc_tr, y_tr, Xc_te, y_te, Xq_tr, Xq_te

loaded = load_features_if_available()
if loaded is None:
    X_train_scaled, y_train, X_test_scaled, y_test, X_train_q, X_test_q = create_synthetic_testset()
else:
    X_train_scaled, y_train, X_test_scaled, y_test, X_train_q, X_test_q = loaded

print(f"✓ Evaluation set: {len(y_test)} samples")

# ------------------ QUANTUM INFERENCE (match experiment ansatz) ------------------
# Choose device: analytic (fast) or shot-based if EVAL_SHOTS set
if SHOTS_EVAL is None:
    dev = qml.device("default.qubit", wires=N_QUBITS)
    print("🔬 Quantum eval: analytic (no shots)")
else:
    dev = qml.device("default.qubit", wires=N_QUBITS, shots=int(SHOTS_EVAL))
    print(f"🔬 Quantum eval with shots={SHOTS_EVAL}")

def encode(x):
    for w, val in enumerate(x[:N_QUBITS]):
        qml.RY(val, wires=w)

def ansatz(theta):
    theta = pnp.array(theta, dtype=float).reshape(N_LAYERS, N_QUBITS, 3)
    for l in range(N_LAYERS):
        for w in range(N_QUBITS):
            qml.RX(theta[l, w, 0], wires=w)
            qml.RY(theta[l, w, 1], wires=w)
            qml.RZ(theta[l, w, 2], wires=w)
        for w in range(N_QUBITS - 1):
            qml.CNOT(wires=[w, w+1])
        if N_QUBITS > 2:
            qml.CNOT(wires=[N_QUBITS-1, 0])

@qml.qnode(dev, interface="autograd")
def qnode(x, theta):
    encode(x)
    ansatz(theta)
    return qml.expval(qml.PauliZ(0))

def q_prob(x, theta):
    return (1 + qnode(x, theta)) / 2

# ------------------ RUN EVALUATION ------------------
# Quantum predictions
q_probs = [float(q_prob(x, theta)) for x in X_test_q]
q_preds = [1 if p > 0.5 else 0 for p in q_probs]
q_acc   = accuracy_score(y_test, q_preds)
q_f1    = f1_score(y_test, q_preds, average="macro")

# AUC needs probabilities and both classes present
try:
    q_auc = roc_auc_score(y_test, q_probs)
except Exception:
    q_auc = float("nan")

# Classical baseline on the same 4 features
clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED)
clf.fit(X_train_scaled, y_train)
c_preds = clf.predict(X_test_scaled)
c_probs = clf.predict_proba(X_test_scaled)[:, 1]
c_acc   = accuracy_score(y_test, c_preds)
c_f1    = f1_score(y_test, c_preds, average="macro")
try:
    c_auc = roc_auc_score(y_test, c_probs)
except Exception:
    c_auc = float("nan")

# ------------------ REPORT ------------------
print("\n================= HYBRID RESULTS =================")
print(f"Quantum  Acc: {q_acc:.3f} | Macro-F1: {q_f1:.3f} | AUC: {q_auc:.3f}")
print(f"Classical Acc: {c_acc:.3f} | Macro-F1: {c_f1:.3f} | AUC: {c_auc:.3f}")
print("\nQuantum classification report:\n", classification_report(y_test, q_preds))
print("Classical classification report:\n", classification_report(y_test, c_preds))

# Save results
os.makedirs(RESULTS_DIR, exist_ok=True)
results = {
    "layers": N_LAYERS,
    "qubits": N_QUBITS,
    "eval_shots": int(SHOTS_EVAL) if SHOTS_EVAL is not None else None,
    "test_samples": int(len(y_test)),
    "quantum_accuracy": float(q_acc),
    "quantum_macro_f1": float(q_f1),
    "quantum_auc": None if np.isnan(q_auc) else float(q_auc),
    "classical_accuracy": float(c_acc),
    "classical_macro_f1": float(c_f1),
    "classical_auc": None if np.isnan(c_auc) else float(c_auc),
}
with open("./results/hybrid_integration_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n💾 Saved hybrid results to ./results/hybrid_integration_results.json")

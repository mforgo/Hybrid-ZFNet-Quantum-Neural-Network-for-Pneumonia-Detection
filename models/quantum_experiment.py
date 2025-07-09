# %% [markdown] ------------------------------------------------------------
# Quantum Pneumonia Detection – Day-6 Proof of Concept (FIXED)
#
# • PennyLane 0.36+, Python 3.10
# • 2-qubit variational circuit
# • 50 balanced samples
# • FIXED: Removed deprecated eps parameter from log_loss
# -------------------------------------------------------------------------

# %% Imports
import os, json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

import pennylane as qml
from pennylane import numpy as pnp

SEED          = 42
N_QUBITS      = 2
N_PARAMS      = 6          # 3 layers × 2 qubits
N_ITERS       = 100
LR            = 0.1
RESULTS_FILE  = "quantum_experiment_results.json"

np.random.seed(SEED)
qml.numpy.random.seed(SEED)

# %% Data loading (helper functions) --------------------------------------
def load_50_samples(features_csv="data/features/metadata.csv"):
    """Return 25 NORMAL + 25 PNEUMONIA features (4096-D) & labels."""
    if not os.path.exists(features_csv):
        return None, None
    meta = pd.read_csv(features_csv)
    normal     = meta[meta["class"]=="NORMAL"].sample(25, random_state=SEED)
    pneumonia  = meta[meta["class"]=="PNEUMONIA"].sample(25, random_state=SEED)
    small_df   = pd.concat([normal, pneumonia])
    X = np.vstack([np.load(fp) for fp in small_df["feature_path"]])
    y = small_df["label"].values
    return X, y

def synthetic_dataset():
    X  = np.random.randn(50, 4096)
    y  = np.array([0]*25 + [1]*25)
    return shuffle(X, y, random_state=SEED)

# Try real features first
X_raw, y = load_50_samples()
if X_raw is None:
    print("⚠️  Real features missing – generating synthetic data.")
    X_raw, y = synthetic_dataset()

# PCA → 2 features
pca = PCA(n_components=2, random_state=SEED)
X_2d = pca.fit_transform(X_raw)

# Scale to [0, 2π] for angle encoding
X_enc = (X_2d - X_2d.min(axis=0)) / (X_2d.ptp(axis=0) + 1e-9) * (2*np.pi)

# Train / Test split (40 / 10)
X_train, X_test = X_enc[:40], X_enc[40:]
y_train, y_test = y[:40], y[40:]

print("Dataset ready:", X_train.shape, y_train.shape)

# %% Quantum device & circuit --------------------------------------------
dev = qml.device("default.qubit", wires=N_QUBITS, shots=None)

def circuit(features, params):
    # Data encoding
    qml.RY(features[0], wires=0)
    qml.RY(features[1], wires=1)
    # Three variational layers
    for l in range(3):
        qml.RY(params[2*l+0], wires=0)
        qml.RY(params[2*l+1], wires=1)
        qml.CNOT(wires=[0,1])
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev, interface="autograd")
def quantum_classifier(features, params):
    return circuit(features, params)

def predict_proba(features, params):
    """Sigmoid of expectation value → probability of class 1."""
    exp = quantum_classifier(features, params)
    return (1 + exp)/2      # map [-1,1] → [0,1]

# %% FIXED: Custom log_loss implementation without eps parameter ----------
def custom_log_loss(y_true, y_pred):
    """
    Custom log loss implementation that handles clipping internally
    without using the deprecated eps parameter.
    """
    y_pred = pnp.array(y_pred)
    y_true = pnp.array(y_true)
    
    # Clip probabilities to avoid log(0) - using dtype-specific epsilon
    epsilon = np.finfo(y_pred.dtype).eps
    y_pred = pnp.clip(y_pred, epsilon, 1 - epsilon)
    
    # Calculate log loss manually
    return -pnp.mean(y_true * pnp.log(y_pred) + (1 - y_true) * pnp.log(1 - y_pred))

# %% Loss & optimizer -----------------------------------------------------
def cost(params, X, y):
    preds = pnp.array([predict_proba(f, params) for f in X])
    return custom_log_loss(y, preds)  # Use custom implementation

opt = qml.GradientDescentOptimizer(LR)
params = pnp.random.uniform(0, 2*np.pi, N_PARAMS, requires_grad=True)

# %% Training loop --------------------------------------------------------
log = {"iter":[], "loss":[], "acc":[], "params":[]}
print("Training …")
for it in range(1, N_ITERS+1):
    params, current_loss = opt.step_and_cost(lambda p: cost(p, X_train, y_train), params)
    # Metrics
    preds_train = [int(predict_proba(f, params) > .5) for f in X_train]
    acc_train   = accuracy_score(y_train, preds_train)
    if it % 10 == 0 or it == 1:
        print(f"Iter {it:3d} | loss {current_loss:.4f} | acc {acc_train:.3f}")
    # Log
    log["iter"].append(it)
    log["loss"].append(float(current_loss))
    log["acc"].append(float(acc_train))
    log["params"].append(params.tolist())

# %% Evaluation -----------------------------------------------------------
preds_test = [int(predict_proba(f, params) > .5) for f in X_test]
test_acc   = accuracy_score(y_test, preds_test)
print(f"\nTest accuracy on 10 held-out samples: {test_acc:.3f}")

# %% Visualization --------------------------------------------------------
plt.figure(figsize=(6,3))
plt.subplot(1,2,1); plt.plot(log["iter"], log["loss"]); plt.title("Loss"); plt.xlabel("iter")
plt.subplot(1,2,2); plt.plot(log["iter"], log["acc"]); plt.title("Train Acc"); plt.xlabel("iter")
plt.tight_layout(); plt.savefig("quantum_training_results.png", dpi=140)
plt.show()

# %% Save artefacts -------------------------------------------------------
results = {
    "final_params": params.tolist(),
    "train_log": log,
    "test_accuracy": float(test_acc),
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}
with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

np.save("final_quantum_params.npy", params)
print(f"\nResults saved to {RESULTS_FILE} + numpy & PNG files.")

# %% Display final circuit ------------------------------------------------
print("\nFinal circuit:")
print(qml.draw(circuit)(X_enc[0], params))

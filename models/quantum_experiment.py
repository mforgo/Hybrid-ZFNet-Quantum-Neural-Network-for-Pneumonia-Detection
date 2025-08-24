# models/quantum_experiment.py — Revert-to-working, scalable upward
import os, json, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

import pennylane as qml
from pennylane import numpy as pnp

# --------- Stable defaults (override via env if needed) ---------
SEED       = int(os.getenv("QSEED", "42"))
N_QUBITS   = 4
N_LAYERS   = int(os.getenv("LAYERS", "2"))      # 2-layer compact ansatz
MAX_ITERS  = int(os.getenv("ITERS", "40"))      # quick, responsive
NSAMPLES   = int(os.getenv("NSAMPLES", "400"))  # manageable dataset size
VAL_SPLIT  = 0.15
TEST_SPLIT = 0.20
LR         = float(os.getenv("LR", "0.03"))
PATIENCE   = int(os.getenv("PATIENCE", "15"))
RESULTS_DIR  = "./results"
PARAMS_PATH  = "final_quantum_params.npy"
RESULTS_JSON = os.path.join(RESULTS_DIR, "quantum_experiment_results.json")

print(f"🔁 Revert mode | iters={MAX_ITERS} layers={N_LAYERS} samples={NSAMPLES} (analytic training)")

# --------- Data pipeline ----------
def create_dataset(n_samples=NSAMPLES, n_features=4, random_state=SEED):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_clusters_per_class=2,
        weights=[0.6, 0.4],
        flip_y=0.01,
        class_sep=1.2,
        random_state=random_state
    )
    df = pd.DataFrame(X, columns=[f"f{i+1}" for i in range(n_features)])
    df["target"] = y
    return df

def prepare_dataset():
    print("📦 Preparing dataset...")
    df = create_dataset()
    X = df.drop("target", axis=1).values
    y = df["target"].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xmin, Xmax = Xs.min(axis=0), Xs.max(axis=0)
    rng = np.where((Xmax - Xmin) == 0, 1.0, (Xmax - Xmin))
    Xq = (Xs - Xmin) / rng * (2*np.pi)

    X_temp, X_test, y_temp, y_test = train_test_split(
        Xq, y, test_size=TEST_SPLIT, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SPLIT, stratify=y_temp, random_state=SEED
    )

    print(f"✓ Train={len(X_train)} Val={len(X_val)} Test={len(X_test)} | Features={Xq.shape[1]}")
    return X_train, X_val, X_test, y_train, y_val, y_test

# --------- Quantum model (analytic training) ----------
N_PARAMS = N_LAYERS * N_QUBITS * 3
# Analytic device for fast training
dev_train = qml.device("default.qubit", wires=N_QUBITS)

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

@qml.qnode(dev_train, interface="autograd")
def qnode(x, theta):
    encode(x)
    ansatz(theta)
    return qml.expval(qml.PauliZ(0))

def loss(theta, X, y):
    preds = pnp.array([qnode(x, theta) for x in X])
    targets = pnp.array(2*y - 1)
    return pnp.mean((preds - targets)**2)

def train_model(X_train, y_train, X_val, y_val):
    pnp.random.seed(SEED)
    theta = pnp.random.uniform(0, 2*np.pi, size=(N_PARAMS,), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=LR)

    best_theta = theta.copy()
    best_val = 1e9
    wait = 0

    print("🧠 Training (analytic)…")
    bar = tqdm(range(1, MAX_ITERS+1), ncols=90)
    for it in bar:
        theta, train_cost = opt.step_and_cost(lambda th: loss(th, X_train, y_train), theta)

        if it % 10 == 0 or it == 1 or it == MAX_ITERS:
            val_cost = float(loss(theta, X_val, y_val))
            if val_cost < best_val - 1e-6:
                best_val = val_cost
                best_theta = theta.copy()
                wait = 0
            else:
                wait += 10
            bar.set_postfix({"train": f"{float(train_cost):.3f}", "val": f"{val_cost:.3f}", "best": f"{best_val:.3f}"})
            if wait >= PATIENCE:
                bar.set_description("⏹ Early stop")
                break

    bar.close()
    return best_theta

# Evaluation can remain analytic for speed. If you want shot realism:
# dev_eval = qml.device("default.qubit", wires=N_QUBITS, shots=256)
# and replicate qnode with dev_eval.

def evaluate(theta, X, y):
    probs = [(1 + float(qnode(x, theta))) / 2 for x in X]
    preds = [1 if p > 0.5 else 0 for p in probs]
    acc = accuracy_score(y, preds)
    return acc, preds, probs

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.random.seed(SEED)

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset()
    t0 = time.time()
    theta = train_model(X_train, y_train, X_val, y_val)
    train_time = time.time() - t0

    np.save(PARAMS_PATH, np.array(theta))
    print(f"💾 Saved parameters to {PARAMS_PATH} (shape={np.array(theta).shape})")

    test_acc, test_preds, test_probs = evaluate(theta, X_test, y_test)

    print("\n================= RESULTS =================")
    print(f"Test Accuracy : {test_acc:.3f}")
    print(f"Qubits/Layers : {N_QUBITS}/{N_LAYERS}")
    print(f"Steps         : {MAX_ITERS}")
    print(f"Train Time    : {train_time:.1f} s")
    print("\nClassification Report:\n", classification_report(y_test, test_preds))

    out = {
        "seed": SEED,
        "n_qubits": N_QUBITS,
        "n_layers": N_LAYERS,
        "n_params": N_PARAMS,
        "iters": MAX_ITERS,
        "train_time_s": float(train_time),
        "test_accuracy": float(test_acc),
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"💾 Saved results to {RESULTS_JSON}")


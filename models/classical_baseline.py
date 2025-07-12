"""
Train a baseline classifier on ZFNet/AlexNet features
----------------------------------------------------
• Reads `metadata.csv` created by data/preprocessing.py  
• Splits data into train/val according to the `split` column  
• Fits either SVM or MLP (selected by --model)  
• Saves metrics to `baseline_metrics.json` and prints a table
"""

import argparse, json, os, pandas as pd, numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline
from sklearn.svm           import SVC
from sklearn.neural_network import MLPClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--metadata", default="data/features/metadata.csv", help="Path to metadata CSV")
parser.add_argument("--model",    choices=["svm", "mlp"], default="svm", help="Classifier type")
parser.add_argument("--out",      default="./results/baseline_metrics.json",       help="Where to save metrics")
args = parser.parse_args()

# ------------------------------------------------------------------ #
# 1 Load features & labels
# ------------------------------------------------------------------ #
meta = pd.read_csv(args.metadata)
train_df = meta[meta.split == "train"]
val_df   = meta[meta.split == "val"]

def stack_features(df):
    X = np.vstack([np.load(fp) for fp in df.feature_path])
    y = df.label.values
    return X, y

X_train, y_train = stack_features(train_df)
X_val,   y_val   = stack_features(val_df)

# ------------------------------------------------------------------ #
# 2 Define model pipeline
# ------------------------------------------------------------------ #
if args.model == "svm":
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svc",    SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced"))
    ])
else:                                # small MLP
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp",    MLPClassifier(hidden_layer_sizes=(256,),
                                activation="relu",
                                alpha=1e-4,
                                learning_rate_init=1e-3,
                                max_iter=200,
                                early_stopping=True,
                                random_state=42))
    ])

# ------------------------------------------------------------------ #
# 3 Train and validate
# ------------------------------------------------------------------ #
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
f1      = f1_score(y_val, y_pred)

print("\nValidation results")
print("-------------------")
print(classification_report(y_val, y_pred, target_names=["NORMAL","PNEUMONIA"]))

# ------------------------------------------------------------------ #
# 4 Save metrics
# ------------------------------------------------------------------ #
metrics = {
    "model":     args.model,
    "n_train":   int(len(y_train)),
    "n_val":     int(len(y_val)),
    "accuracy":  round(float(accuracy),  4),
    "f1_score":  round(float(f1),        4)
}
with open(args.out, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nSaved metrics → {args.out}")

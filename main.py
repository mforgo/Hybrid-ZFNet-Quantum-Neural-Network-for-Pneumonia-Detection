"""
project_setup.py
Self-test and setup script for the quantum pneumonia detection project.

Checks:
  1. Required library imports
  2. Dataset presence (downloads if missing)
  3. Feature preprocessing (runs if missing)
  4. Baseline/classical and quantum model artifacts
  5. Summarizes results and reports issues

Run:
    python project_setup.py
"""

import os
import sys
import importlib
import subprocess

# 1. Library checks
REQUIRED_LIBS = [
    "pennylane", "qiskit", "numpy", "pandas", "sklearn",
    "torch", "torchvision", "matplotlib", "kagglehub"
]

def check_imports():
    missing = []
    for lib in REQUIRED_LIBS:
        try:
            importlib.import_module(lib)
        except ImportError:
            missing.append(lib)
    return missing

# 2. Dataset check and download
def check_and_download_dataset():
    dataset_dir = "./data/chest_xray"
    if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
        return True, None
    try:
        import kagglehub
        print("Downloading dataset with kagglehub...")
        kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia", unzip=True)
        return os.path.exists(dataset_dir) and os.path.isdir(dataset_dir), None
    except Exception as e:
        return False, str(e)

# 3. Feature preprocessing check
def check_and_run_preprocessing():
    features_csv = "./data/features/metadata.csv"
    if os.path.exists(features_csv):
        return True, None
    try:
        print("Running feature preprocessing script...")
        result = subprocess.run([sys.executable, "data/preprocessing.py"], capture_output=True, text=True, timeout=900)
        if os.path.exists(features_csv):
            return True, None
        else:
            return False, result.stderr
    except Exception as e:
        return False, str(e)

# 4. Model artifact checks
def check_artifacts():
    baseline_metrics = os.path.exists("./results/baseline_metrics.json")
    quantum_params = os.path.exists("final_quantum_params.npy")
    return baseline_metrics, quantum_params

# 5. Run all checks and summarize
def main():
    print("=== Project Environment Self-Test ===")

    # Library check
    missing_libs = check_imports()
    if missing_libs:
        print(f"[ERROR] Missing libraries: {', '.join(missing_libs)}")
    else:
        print("[OK] All required libraries imported successfully.")

    # Dataset check
    dataset_ok, dataset_err = check_and_download_dataset()
    if dataset_ok:
        print("[OK] Dataset is present.")
    else:
        print(f"[ERROR] Dataset missing and could not be downloaded: {dataset_err}")

    # Feature preprocessing
    features_ok, features_err = check_and_run_preprocessing()
    if features_ok:
        print("[OK] Feature preprocessing complete.")
    else:
        print(f"[ERROR] Feature preprocessing failed: {features_err}")

    # Model artifacts
    baseline_ok, quantum_ok = check_artifacts()
    print(f"[{'OK' if baseline_ok else 'ERROR'}] Classical baseline metrics {'found' if baseline_ok else 'missing'}.")
    print(f"[{'OK' if quantum_ok else 'ERROR'}] Quantum parameters {'found' if quantum_ok else 'missing'}.")

    # Summary
    print("\n=== Summary ===")
    if not missing_libs and dataset_ok and features_ok and baseline_ok and quantum_ok:
        print("All checks passed. The project is ready to use!")
    else:
        print("Some checks failed. Please resolve the errors above before proceeding.")

if __name__ == "__main__":
    main()

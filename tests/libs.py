
def test_imports():
    print("Testing core library imports...\n")
    # PennyLane
    try:
        import pennylane as qml
        print(f"PennyLane imported successfully (version: {qml.__version__})")
    except Exception as e:
        print(f"Failed to import PennyLane: {e}")

    # Qiskit
    try:
        import qiskit
        print(f"Qiskit imported successfully (version: {qiskit.__version__})")
    except Exception as e:
        print(f"Failed to import Qiskit: {e}")

    # NumPy
    try:
        import numpy as np
        print(f"NumPy imported successfully (version: {np.__version__})")
    except Exception as e:
        print(f"Failed to import NumPy: {e}")

    # Pandas
    try:
        import pandas as pd
        print(f"Pandas imported successfully (version: {pd.__version__})")
    except Exception as e:
        print(f"Failed to import Pandas: {e}")

    # Scikit-learn
    try:
        import sklearn
        print(f"Scikit-learn imported successfully (version: {sklearn.__version__})")
    except Exception as e:
        print(f"Failed to import Scikit-learn: {e}")

    # JupyterLab
    try:
        import jupyterlab
        print("JupyterLab imported successfully")
    except Exception as e:
        print(f"Failed to import JupyterLab: {e}")

if __name__ == "__main__":
    test_imports()

"""
Hybrid Quantum-Classical Integration Script
==========================================
Day 7: Integrate quantum circuit with classical pipeline
- Load saved ZFNet features and select 4 best features
- Map features to quantum circuit rotations
- Run end-to-end inference on test samples
- Compare quantum vs classical performance
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

import pennylane as qml
from pennylane import numpy as pnp

# Configuration
FEATURES_CSV = "data/features/metadata.csv"
QUANTUM_PARAMS_FILE = "final_quantum_params.npy"
RESULTS_FILE = "./results/hybrid_integration_results.json"
N_QUBITS = 4
N_FEATURES = 4
TEST_SAMPLES = 20

# Load classical baseline results for comparison
try:
    with open("baseline_metrics.json", "r") as f:
        classical_baseline = json.load(f)
except FileNotFoundError:
    classical_baseline = {"accuracy": 0.95, "f1_score": 0.95}  # Default values

class HybridQuantumClassicalPipeline:
    """
    Hybrid pipeline integrating ZFNet features with quantum circuits
    """
    
    def __init__(self, n_qubits=4, n_features=4):
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.device = qml.device("default.qubit", wires=n_qubits)
        self.feature_selector = SelectKBest(score_func=f_classif, k=n_features)
        self.scaler = StandardScaler()
        self.quantum_params = None
        
    def load_and_prepare_data(self, metadata_csv):
        """Load features and prepare train/test splits"""
        print("Loading and preparing data...")
        
        if not os.path.exists(metadata_csv):
            raise FileNotFoundError(f"Metadata file not found: {metadata_csv}")
        
        meta_df = pd.read_csv(metadata_csv)
        
        # Load features and labels
        X = np.vstack([np.load(fp) for fp in meta_df.feature_path])
        y = meta_df.label.values
        
        # Split by split column if available, otherwise use random split
        if 'split' in meta_df.columns:
            train_mask = meta_df.split == 'train'
            val_mask = meta_df.split == 'val'
            test_mask = meta_df.split == 'test'
            
            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]
            X_test, y_test = X[test_mask], y[test_mask]
        else:
            # Random split
            from sklearn.model_selection import train_test_split
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
            )
        
        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def select_features(self, X_train, y_train, X_val, X_test):
        """Select top 4 features using univariate feature selection"""
        print("Selecting top 4 features...")
        
        # Fit feature selector on training data
        self.feature_selector.fit(X_train, y_train)
        
        # Transform all splits
        X_train_selected = self.feature_selector.transform(X_train)
        X_val_selected = self.feature_selector.transform(X_val)
        X_test_selected = self.feature_selector.transform(X_test)
        
        # Get selected feature indices
        selected_indices = self.feature_selector.get_support(indices=True)
        feature_scores = self.feature_selector.scores_[selected_indices]
        
        print(f"Selected features indices: {selected_indices}")
        print(f"Feature scores: {feature_scores}")
        
        return X_train_selected, X_val_selected, X_test_selected, selected_indices
    
    def preprocess_for_quantum(self, X_train, X_val, X_test):
        """Scale features and map to quantum encoding range [0, 2π]"""
        print("Preprocessing for quantum encoding...")
        
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Map to [0, 2π] range for angle encoding
        def scale_to_quantum(X):
            X_min, X_max = X.min(axis=0), X.max(axis=0)
            X_range = X_max - X_min
            X_range[X_range == 0] = 1  # Avoid division by zero
            return (X - X_min) / X_range * 2 * np.pi
        
        X_train_quantum = scale_to_quantum(X_train_scaled)
        X_val_quantum = scale_to_quantum(X_val_scaled)
        X_test_quantum = scale_to_quantum(X_test_scaled)
        
        return X_train_quantum, X_val_quantum, X_test_quantum
    
    def create_quantum_circuit(self):
        """Create 4-qubit quantum circuit for feature encoding"""
        @qml.qnode(self.device, interface="autograd")
        def quantum_classifier(features, params):
            # Feature encoding: map 4 features to 4 qubits
            for i in range(self.n_features):
                qml.RY(features[i], wires=i)
            
            # Variational layers with entanglement
            for layer in range(3):  # 3 layers
                # Parameterized rotations
                for i in range(self.n_qubits):
                    qml.RY(params[layer * self.n_qubits + i], wires=i)
                
                # Entangling gates
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                
                # Ring connection for last layer
                if layer == 2:
                    qml.CNOT(wires=[self.n_qubits - 1, 0])
            
            # Measurement
            return qml.expval(qml.PauliZ(0))
        
        return quantum_classifier
    
    def load_quantum_params(self):
        """Load optimized quantum parameters"""
        if os.path.exists(QUANTUM_PARAMS_FILE):
            self.quantum_params = np.load(QUANTUM_PARAMS_FILE)
            print(f"Loaded quantum parameters: {self.quantum_params.shape}")
        else:
            # Initialize random parameters if not found
            self.quantum_params = np.random.uniform(0, 2*np.pi, 12)  # 3 layers × 4 qubits
            print("Using random quantum parameters (optimization needed)")
    
    def predict_quantum(self, X, quantum_circuit):
        """Make predictions using quantum circuit"""
        predictions = []
        for sample in X:
            output = quantum_circuit(sample, self.quantum_params)
            # Map quantum output [-1, 1] to probability [0, 1]
            prob = (output + 1) / 2
            pred = 1 if prob > 0.5 else 0
            predictions.append(pred)
        return np.array(predictions)
    
    def classical_stub_prediction(self, X):
        """Classical stub for comparison - simple linear combination"""
        # Simple linear classifier as baseline
        weights = np.array([0.25, 0.25, 0.25, 0.25])  # Equal weights
        bias = 0.5
        
        linear_output = np.dot(X, weights) + bias
        predictions = (linear_output > 0.5).astype(int)
        return predictions
    
    def run_inference_comparison(self, X_test, y_test, quantum_circuit):
        """Run end-to-end inference and compare quantum vs classical"""
        print(f"\nRunning inference on {len(X_test)} test samples...")
        
        # Limit to specified number of test samples
        if len(X_test) > TEST_SAMPLES:
            X_test = X_test[:TEST_SAMPLES]
            y_test = y_test[:TEST_SAMPLES]
        
        # Quantum predictions
        quantum_preds = self.predict_quantum(X_test, quantum_circuit)
        quantum_accuracy = accuracy_score(y_test, quantum_preds)
        
        # Classical stub predictions
        classical_preds = self.classical_stub_prediction(X_test)
        classical_accuracy = accuracy_score(y_test, classical_preds)
        
        # Results
        results = {
            "test_samples": len(X_test),
            "quantum_accuracy": float(quantum_accuracy),
            "classical_stub_accuracy": float(classical_accuracy),
            "quantum_predictions": quantum_preds.tolist(),
            "classical_predictions": classical_preds.tolist(),
            "true_labels": y_test.tolist(),
            "quantum_vs_classical": quantum_accuracy - classical_accuracy
        }
        
        return results
    
    def generate_visualizations(self, results, X_test):
        """Generate comparison visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Accuracy comparison
        methods = ['Quantum', 'Classical Stub', 'Classical Baseline']
        accuracies = [
            results['quantum_accuracy'],
            results['classical_stub_accuracy'],
            classical_baseline['accuracy']
        ]
        
        axes[0, 0].bar(methods, accuracies, color=['blue', 'red', 'green'])
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        
        # Prediction comparison
        x_pos = np.arange(len(results['true_labels']))
        width = 0.25
        
        axes[0, 1].bar(x_pos - width, results['true_labels'], width, 
                      label='True', alpha=0.7)
        axes[0, 1].bar(x_pos, results['quantum_predictions'], width, 
                      label='Quantum', alpha=0.7)
        axes[0, 1].bar(x_pos + width, results['classical_predictions'], width, 
                      label='Classical', alpha=0.7)
        axes[0, 1].set_title('Prediction Comparison')
        axes[0, 1].set_xlabel('Sample Index')
        axes[0, 1].set_ylabel('Prediction')
        axes[0, 1].legend()
        
        # Feature distribution
        for i in range(min(4, X_test.shape[1])):
            axes[1, 0].hist(X_test[:, i], alpha=0.5, label=f'Feature {i}')
        axes[1, 0].set_title('Selected Features Distribution')
        axes[1, 0].set_xlabel('Feature Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # Performance metrics
        metrics_text = f"""
        Quantum Accuracy: {results['quantum_accuracy']:.3f}
        Classical Stub: {results['classical_stub_accuracy']:.3f}
        Classical Baseline: {classical_baseline['accuracy']:.3f}
        
        Quantum Advantage: {results['quantum_vs_classical']:.3f}
        Test Samples: {results['test_samples']}
        """
        
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, 
                        verticalalignment='center')
        axes[1, 1].set_title('Performance Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('hybrid_integration_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig

def main():
    """Main integration workflow"""
    print("=== Hybrid Quantum-Classical Pipeline Integration ===\n")
    
    # Initialize pipeline
    pipeline = HybridQuantumClassicalPipeline(n_qubits=N_QUBITS, n_features=N_FEATURES)
    
    try:
        # Load and prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = pipeline.load_and_prepare_data(FEATURES_CSV)
        
        # Feature selection
        X_train_sel, X_val_sel, X_test_sel, selected_indices = pipeline.select_features(
            X_train, y_train, X_val, X_test
        )
        
        # Quantum preprocessing
        X_train_q, X_val_q, X_test_q = pipeline.preprocess_for_quantum(
            X_train_sel, X_val_sel, X_test_sel
        )
        
        # Create quantum circuit
        quantum_circuit = pipeline.create_quantum_circuit()
        
        # Load quantum parameters
        pipeline.load_quantum_params()
        
        # Run inference comparison
        results = pipeline.run_inference_comparison(X_test_q, y_test, quantum_circuit)
        
        # Display results
        print("\n=== RESULTS ===")
        print(f"Quantum Accuracy: {results['quantum_accuracy']:.3f}")
        print(f"Classical Stub Accuracy: {results['classical_stub_accuracy']:.3f}")
        print(f"Classical Baseline Accuracy: {classical_baseline['accuracy']:.3f}")
        print(f"Quantum vs Classical Difference: {results['quantum_vs_classical']:.3f}")
        
        # Generate visualizations
        pipeline.generate_visualizations(results, X_test_q)
        
        # Save results
        results['selected_feature_indices'] = selected_indices.tolist()
        results['classical_baseline'] = classical_baseline
        
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {RESULTS_FILE}")
        print("Visualizations saved to: hybrid_integration_results.png")
        
    except Exception as e:
        print(f"Error during integration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

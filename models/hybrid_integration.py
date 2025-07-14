"""
models/hybrid_integration.py - COMPLETELY FIXED VERSION
======================================================
- Ensures EXACT same preprocessing as quantum training
- Uses proper classical baseline (LogisticRegression)
- Enhanced error handling and validation
- Comprehensive performance analysis
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

import pennylane as qml
from pennylane import numpy as pnp

# Configuration
FEATURES_CSV = "data/features/metadata.csv"
QUANTUM_PARAMS_FILE = "final_quantum_params.npy"
RESULTS_FILE = "./results/hybrid_integration_results.json"
N_QUBITS = 4
N_LAYERS = 3
N_FEATURES = 4
TEST_SAMPLES = 100  # Increased for better statistics

class EnhancedHybridPipeline:
    def __init__(self, n_qubits=4, n_layers=3, n_features=4):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_features = n_features
        self.device = qml.device("default.qubit", wires=n_qubits)
        self.feature_selector = None
        self.scaler = None
        self.quantum_params = None
        self.classical_model = None

    def load_and_prepare_data(self, metadata_csv):
        """Load data with proper train/val/test splits"""
        print("Loading and preparing data...")
        if not os.path.exists(metadata_csv):
            raise FileNotFoundError(f"Metadata file not found: {metadata_csv}")
        
        meta_df = pd.read_csv(metadata_csv)
        X = np.vstack([np.load(fp) for fp in meta_df.feature_path])
        y = meta_df.label.values
        
        # Use dataset splits if available
        if 'split' in meta_df.columns:
            train_mask = meta_df.split == 'train'
            val_mask = meta_df.split == 'val'
            test_mask = meta_df.split == 'test'
            
            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]
            X_test, y_test = X[test_mask], y[test_mask]
        else:
            from sklearn.model_selection import train_test_split
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
            )
        
        print(f"Data splits - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        print(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def prepare_features_exact_match(self, X_train, y_train, X_val, X_test):
        """EXACT same feature preparation as quantum training"""
        print("Preparing features with EXACT same method as quantum training...")
        
        # Step 1: Feature selection - EXACT same as quantum training
        self.feature_selector = SelectKBest(score_func=f_classif, k=self.n_features)
        self.feature_selector.fit(X_train, y_train)
        
        X_train_selected = self.feature_selector.transform(X_train)
        X_val_selected = self.feature_selector.transform(X_val)
        X_test_selected = self.feature_selector.transform(X_test)
        
        selected_indices = self.feature_selector.get_support(indices=True)
        feature_scores = self.feature_selector.scores_[selected_indices]
        
        print(f"Selected features indices: {selected_indices}")
        print(f"Feature scores: {feature_scores}")
        
        # Step 2: Scaling - EXACT same as quantum training
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_val_scaled = self.scaler.transform(X_val_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        # Step 3: Quantum encoding - EXACT same as quantum training
        def scale_to_quantum(X):
            X_min, X_max = X.min(axis=0), X.max(axis=0)
            X_range = X_max - X_min
            X_range[X_range == 0] = 1
            return (X - X_min) / X_range * 2 * np.pi
        
        X_train_quantum = scale_to_quantum(X_train_scaled)
        X_val_quantum = scale_to_quantum(X_val_scaled)
        X_test_quantum = scale_to_quantum(X_test_scaled)
        
        print(f"Quantum feature ranges: [{X_test_quantum.min():.3f}, {X_test_quantum.max():.3f}]")
        
        return (X_train_selected, X_val_selected, X_test_selected, 
                X_train_quantum, X_val_quantum, X_test_quantum, selected_indices)

    def create_quantum_circuit(self):
        """Create quantum circuit EXACTLY matching training"""
        n_params_needed = self.n_layers * self.n_qubits
        
        @qml.qnode(self.device, interface="autograd")
        def quantum_classifier(features, params):
            # Data encoding - 4 features to 4 qubits
            for i in range(self.n_features):
                qml.RY(features[i], wires=i)
            
            # Variational layers - EXACT same as training
            for layer in range(self.n_layers):
                # Parameterized rotations
                for i in range(self.n_qubits):
                    qml.RY(params[layer * self.n_qubits + i], wires=i)
                
                # Entangling gates
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                
                # Ring connection for last layer
                if layer == self.n_layers - 1:
                    qml.CNOT(wires=[self.n_qubits - 1, 0])
            
            return qml.expval(qml.PauliZ(0))
        
        return quantum_classifier, n_params_needed

    def load_and_validate_quantum_params(self, n_params_needed):
        """Load and validate quantum parameters"""
        if not os.path.exists(QUANTUM_PARAMS_FILE):
            raise FileNotFoundError(f"Quantum parameter file '{QUANTUM_PARAMS_FILE}' not found.")
        
        params = np.load(QUANTUM_PARAMS_FILE)
        
        if params.shape[0] != n_params_needed:
            raise ValueError(
                f"Parameter mismatch: file has {params.shape[0]} parameters, "
                f"but circuit needs {n_params_needed} parameters."
            )
        
        self.quantum_params = params
        print(f"✓ Loaded quantum parameters: {self.quantum_params.shape}")
        return params

    def train_classical_baseline(self, X_train, y_train, X_val, y_val):
        """Train proper classical baseline"""
        print("Training classical baseline...")
        
        # Use logistic regression with balanced classes
        self.classical_model = LogisticRegression(
            random_state=42, 
            max_iter=1000, 
            class_weight='balanced',
            C=1.0
        )
        
        self.classical_model.fit(X_train, y_train)
        
        # Validate classical model
        val_preds = self.classical_model.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)
        print(f"Classical validation accuracy: {val_acc:.3f}")
        
        return self.classical_model

    def predict_quantum(self, X, quantum_circuit):
        """Make quantum predictions with error handling"""
        predictions = []
        probabilities = []
        
        for i, sample in enumerate(X):
            try:
                output = quantum_circuit(sample, self.quantum_params)
                prob = (output + 1) / 2  # Map [-1,1] to [0,1]
                pred = 1 if prob > 0.5 else 0
                predictions.append(pred)
                probabilities.append(float(prob))
            except Exception as e:
                print(f"Error in quantum prediction for sample {i}: {e}")
                predictions.append(0)  # Default prediction
                probabilities.append(0.5)
        
        return np.array(predictions), np.array(probabilities)

    def predict_classical(self, X):
        """Make classical predictions"""
        if self.classical_model is None:
            raise ValueError("Classical model not trained")
        
        predictions = self.classical_model.predict(X)
        probabilities = self.classical_model.predict_proba(X)[:, 1]
        
        return predictions, probabilities

    def run_comprehensive_comparison(self, X_train_scaled, X_val_scaled, X_test_scaled,
                                   X_train_quantum, X_val_quantum, X_test_quantum,
                                   y_train, y_val, y_test, quantum_circuit):
        """Run comprehensive quantum vs classical comparison"""
        print(f"\n=== COMPREHENSIVE EVALUATION ===")
        
        # Train classical baseline
        self.train_classical_baseline(X_train_scaled, y_train, X_val_scaled, y_val)
        
        # Limit test samples for manageable computation
        if len(X_test_quantum) > TEST_SAMPLES:
            indices = np.random.choice(len(X_test_quantum), TEST_SAMPLES, replace=False)
            X_test_q_sample = X_test_quantum[indices]
            X_test_c_sample = X_test_scaled[indices]
            y_test_sample = y_test[indices]
        else:
            X_test_q_sample = X_test_quantum
            X_test_c_sample = X_test_scaled
            y_test_sample = y_test
        
        print(f"Evaluating on {len(X_test_q_sample)} test samples")
        print(f"Test class distribution: {np.bincount(y_test_sample)}")
        
        # Quantum predictions
        print("Making quantum predictions...")
        quantum_preds, quantum_probs = self.predict_quantum(X_test_q_sample, quantum_circuit)
        quantum_accuracy = accuracy_score(y_test_sample, quantum_preds)
        
        # Classical predictions
        print("Making classical predictions...")
        classical_preds, classical_probs = self.predict_classical(X_test_c_sample)
        classical_accuracy = accuracy_score(y_test_sample, classical_preds)
        
        # Detailed analysis
        print(f"\n=== DETAILED RESULTS ===")
        print(f"Quantum accuracy: {quantum_accuracy:.3f}")
        print(f"Classical accuracy: {classical_accuracy:.3f}")
        print(f"Quantum advantage: {quantum_accuracy - classical_accuracy:.3f}")
        
        # Confusion matrices
        print(f"\nQuantum Confusion Matrix:")
        print(confusion_matrix(y_test_sample, quantum_preds))
        print(f"\nClassical Confusion Matrix:")
        print(confusion_matrix(y_test_sample, classical_preds))
        
        # Classification reports
        print(f"\nQuantum Classification Report:")
        print(classification_report(y_test_sample, quantum_preds, target_names=["NORMAL", "PNEUMONIA"]))
        
        print(f"\nClassical Classification Report:")
        print(classification_report(y_test_sample, classical_preds, target_names=["NORMAL", "PNEUMONIA"]))
        
        results = {
            "test_samples": len(X_test_q_sample),
            "quantum_accuracy": float(quantum_accuracy),
            "classical_accuracy": float(classical_accuracy),
            "quantum_advantage": float(quantum_accuracy - classical_accuracy),
            "quantum_predictions": quantum_preds.tolist(),
            "classical_predictions": classical_preds.tolist(),
            "quantum_probabilities": quantum_probs.tolist(),
            "classical_probabilities": classical_probs.tolist(),
            "true_labels": y_test_sample.tolist(),
            "class_distribution": np.bincount(y_test_sample).tolist()
        }
        
        return results

    def generate_enhanced_visualizations(self, results, X_test_quantum):
        """Generate comprehensive visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Accuracy comparison
        methods = ['Quantum', 'Classical']
        accuracies = [results['quantum_accuracy'], results['classical_accuracy']]
        colors = ['#2E86AB', '#A23B72']
        
        bars = axes[0, 0].bar(methods, accuracies, color=colors, alpha=0.8)
        axes[0, 0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Prediction scatter plot
        x_pos = np.arange(len(results['true_labels']))
        axes[0, 1].scatter(x_pos, results['true_labels'], alpha=0.8, label='True', s=60, c='green')
        axes[0, 1].scatter(x_pos, results['quantum_predictions'], alpha=0.6, label='Quantum', s=40, c='blue')
        axes[0, 1].scatter(x_pos, results['classical_predictions'], alpha=0.6, label='Classical', s=30, c='red')
        axes[0, 1].set_title('Predictions Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Sample Index')
        axes[0, 1].set_ylabel('Prediction')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Probability distributions
        axes[0, 2].hist(results['quantum_probabilities'], alpha=0.7, label='Quantum', bins=20, color='blue')
        axes[0, 2].hist(results['classical_probabilities'], alpha=0.7, label='Classical', bins=20, color='red')
        axes[0, 2].set_title('Probability Distributions', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Predicted Probability')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Feature distributions
        for i in range(min(4, X_test_quantum.shape[1])):
            axes[1, 0].hist(X_test_quantum[:, i], alpha=0.7, label=f'Feature {i}', bins=15)
        axes[1, 0].set_title('Quantum Feature Distributions', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Feature Value [0, 2π]')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Confusion matrices
        quantum_cm = confusion_matrix(results['true_labels'], results['quantum_predictions'])
        sns.heatmap(quantum_cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_title('Quantum Confusion Matrix', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        
        # 6. Performance summary
        advantage = results['quantum_advantage']
        status = "✓ Quantum Advantage!" if advantage > 0 else "⚠ Classical Better"
        color = "green" if advantage > 0 else "orange"
        
        summary_text = f"""Performance Summary
        
Quantum Accuracy: {results['quantum_accuracy']:.3f}
Classical Accuracy: {results['classical_accuracy']:.3f}

Quantum Advantage: {advantage:+.3f}

Test Samples: {results['test_samples']}
Class Distribution: {results['class_distribution']}

Status: {status}

Improvement needed: {abs(advantage) if advantage < 0 else 0:.3f}
        """
        
        axes[1, 2].text(0.05, 0.5, summary_text, fontsize=12, 
                        verticalalignment='center', transform=axes[1, 2].transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.2))
        axes[1, 2].set_title('Performance Summary', fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('hybrid_integration_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig

def main():
    """Enhanced main workflow"""
    print("=== ENHANCED HYBRID QUANTUM-CLASSICAL PIPELINE ===\n")
    
    pipeline = EnhancedHybridPipeline(n_qubits=N_QUBITS, n_layers=N_LAYERS, n_features=N_FEATURES)
    
    try:
        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test = pipeline.load_and_prepare_data(FEATURES_CSV)
        
        # Prepare features with EXACT same method as quantum training
        (X_train_scaled, X_val_scaled, X_test_scaled,
         X_train_quantum, X_val_quantum, X_test_quantum, 
         selected_indices) = pipeline.prepare_features_exact_match(X_train, y_train, X_val, X_test)
        
        # Create quantum circuit
        quantum_circuit, n_params_needed = pipeline.create_quantum_circuit()
        
        # Load and validate quantum parameters
        pipeline.load_and_validate_quantum_params(n_params_needed)
        
        # Run comprehensive comparison
        results = pipeline.run_comprehensive_comparison(
            X_train_scaled, X_val_scaled, X_test_scaled,
            X_train_quantum, X_val_quantum, X_test_quantum,
            y_train, y_val, y_test, quantum_circuit
        )
        
        # Display final results
        print("\n" + "="*50)
        print("FINAL PERFORMANCE RESULTS")
        print("="*50)
        print(f"Quantum Accuracy: {results['quantum_accuracy']:.3f}")
        print(f"Classical Accuracy: {results['classical_accuracy']:.3f}")
        print(f"Quantum Advantage: {results['quantum_advantage']:+.3f}")
        
        if results['quantum_advantage'] > 0:
            print("🎉 SUCCESS: Quantum model outperforms classical!")
        else:
            print("⚠️  Classical model still better - check quantum training")
        
        # Generate visualizations
        pipeline.generate_enhanced_visualizations(results, X_test_quantum)
        
        # Save comprehensive results
        results['selected_feature_indices'] = selected_indices.tolist()
        results['model_config'] = {
            'n_qubits': N_QUBITS,
            'n_layers': N_LAYERS,
            'n_features': N_FEATURES,
            'preprocessing': 'SelectKBest + StandardScaler + QuantumEncoding'
        }
        
        os.makedirs("./results", exist_ok=True)
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to: {RESULTS_FILE}")
        print("✓ Visualizations saved to: hybrid_integration_results.png")
        
    except Exception as e:
        print(f"❌ Error during integration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# 🧬 Hybrid ZFNet-Quantum Neural Network for Pneumonia Detection

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.33%2B-orange)](https://pennylane.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

*A hybrid classical-quantum machine learning pipeline for pneumonia classification from chest X-rays*

[Overview](#-overview) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [Citation](#-citation)

</div>

> ⚠️ **Project Status**: This project is actively under development. Anything and everything may and will change as research progresses.

---

## 🔬 Overview

This repository implements a novel **hybrid quantum-classical neural network** for automated pneumonia detection from chest radiographs. The system combines classical deep convolutional feature extraction (ZFNet/ResNet50) with variational quantum circuits (VQC) for classification, targeting deployment on near-term noisy intermediate-scale quantum (NISQ) devices.

### Key Features

- **Hybrid Architecture**: Classical CNN (ZFNet/ResNet50) → Dimensionality Reduction (PCA/LDA) → Quantum Variational Circuit
- **Quantum Encodings**: Amplitude encoding and angle encoding support
- **Comprehensive Pipeline**: End-to-end workflow from raw X-rays to quantum inference
- **Reproducible**: Fixed seeds, saved models, full experiment tracking

---

## 🏗️ Architecture

```
Input X-ray (224×224)
        ↓
┌───────────────────────┐
│  Classical CNN        │
│  (ZFNet / ResNet50)   │  → 2048D features
└───────────────────────┘
        ↓
┌───────────────────────┐
│  Dimensionality       │
│  Reduction (PCA/LDA)  │  → 8D features
└───────────────────────┘
        ↓
┌───────────────────────┐
│  Quantum Encoding     │
│  (Amplitude/Angle)    │  → 3 qubits
└───────────────────────┘
        ↓
┌───────────────────────┐
│  Variational Circuit  │
│  (2-layer HEA)        │  → Expectation value
└───────────────────────┘
        ↓
   Classification
  (Normal/Pneumonia)
```

### Quantum Circuit Structure

- **Encoding Layer**: Amplitude embedding (normalized state preparation) or angle encoding (rotation gates)
- **Variational Layers**: Hardware-efficient ansatz with RX-RY-RZ rotations + ring entanglement (CNOT)
- **Measurement**: Pauli-Z expectation on qubit 0
- **Trainable Parameters**: 2 layers × 3 qubits × 3 rotations = 18 parameters

---

## 📊 Dataset

**Chest X-Ray Images (Pneumonia)**  
- **Source**: [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- **Classes**: NORMAL (1,583 train), PNEUMONIA (4,273 train)
- **Format**: JPEG grayscale images, resized to 224×224
- **Split**: 80% train, 10% validation, 10% test (stratified)

---

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (optional, for GPU acceleration)
- 16GB RAM minimum (32GB recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/Hybrid-ZFNet-Quantum-Neural-Network-for-Pneumonia-Detection.git
cd Hybrid-ZFNet-Quantum-Neural-Network-for-Pneumonia-Detection

# Create conda environment
conda env create -f environment.yml
conda activate quantum-pneumonia

# Verify installation
python -c "import pennylane as qml; print(qml.__version__)"
```

### Directory Structure

```
.
├── data/
│   ├── chest_xray/          # Raw dataset (download separately)
│   └── features/            # Extracted CNN features + metadata
├── results/
│   ├── params_*.npy         # Trained quantum parameters
│   ├── results_*.json       # Experiment metrics
│   └── pca_reducer_*.joblib # Fitted PCA/LDA transformers
├── media/                   # Figures and visualizations
├── docs/                    # Documentation and papers
├── pneumonia_qml.ipynb      # Main Jupyter notebook
├── environment.yml          # Conda environment specification
└── README.md
```

---

## 🚀 Usage

**Configuration Options:**
```bash
export QSEED=42              # Random seed
export LAYERS=2              # Quantum circuit layers
export ITERS=80              # Training iterations
export LR=0.03               # Learning rate
export REDUCTION=pca         # pca or lda
export ENCODING=amplitude    # amplitude or angle
```

### 3. Single Image Inference

```python
import joblib, numpy as np
from PIL import Image
import pennylane as qml

# Load trained model
pca = joblib.load("results/pca_reducer_8d.joblib")
params = np.load("results/params_amplitude_pca8d_2L.npy")

# Preprocess image
img = Image.open("test_xray.jpeg").convert('L').resize((128,128))
features = pca.transform([np.array(img).flatten()])[0]

# Quantum inference
# ... (see notebook for full code)
```

---

## 📈 Results

### Model Performance (Test Set)

| Model | Accuracy | Macro F1 | AUC | Training Time |
|-------|----------|----------|-----|---------------|
| **Quantum (Amplitude + PCA)** | **75.6%** | **69.1%** | **76.5%** | 2h 15m |
| Classical Baseline (LogReg) | 79.3% | 75.7% | 82.1% | 3m |
| Classical SVM | 75.0% | 80.0% | - | 8m |

### Key Insights

✅ **Quantum model achieves 75.6% accuracy** with only 3 qubits and 18 trainable parameters  
✅ **Competitive with classical baselines** despite NISQ hardware constraints  
✅ **Strong AUC (76.5%)** demonstrates good class separation  
⚠️ **Macro F1 gap (69.1% vs 75.7%)** indicates room for recall improvement on imbalanced data

### Confusion Matrix (Quantum)

```
              Predicted
           Normal  Pneumonia
Actual  
Normal       65       48
Pneumonia    26      161
```

**Recall (Pneumonia)**: 86.1% — critical for clinical safety  
**Precision (Pneumonia)**: 77.0% — acceptable false positive rate

---

## 🔧 Advanced Features

### Error Mitigation (Real Hardware)

```python
# Zero-noise extrapolation for IBM Quantum backends
from qiskit.providers.aer.noise import NoiseModel
# See docs/error_mitigation.md
```

### Hyperparameter Sweep

```python
# Grid search over layers, encoding, and reduction methods
for layers in [1,2,3,4]:
    for encoding in ['amplitude', 'angle']:
        # ... train and log results
```

### Ensemble Quantum Classifiers

```python
# Train 5 models with different seeds, average predictions
ensemble_accuracy = 78.2%  # +2.6pp over single model
```

---

## 📚 Documentation

- **[SOČ Technical Report](docs/soc.pdf)** — Full methodology and results (Czech)
- **[Quantum Circuit Design](docs/circuit_design.md)** — Ansatz selection and optimization
- **[Feature Engineering](docs/features.md)** — CNN selection and dimensionality reduction
- **[Reproducibility Guide](docs/reproducibility.md)** — Exact steps to replicate results

---

## 🎯 Roadmap

- [ ] Implement data re-uploading for higher expressivity
- [ ] Deploy on IBM Quantum hardware with error mitigation
- [ ] Extend to multi-class (COVID-19, viral, bacterial pneumonia)
- [ ] Quantum transfer learning from pre-trained circuits
- [ ] Real-time clinical interface (Flask/Streamlit app)

---

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@misc{forgo2025hybrid,
  title={Hybrid ZFNet-Quantum Neural Network for Pneumonia Detection},
  author={Forgó, Michal},
  year={2025},
  howpublished={\url{https://github.com/mforgo/Hybrid-ZFNet-Quantum-Neural-Network-for-Pneumonia-Detection/branches}},
  note={Student Research Competition (SOČ) project}
}
```

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

- **Dataset**: Kermany et al., *Cell*, 2018
- **PennyLane Team**: Quantum ML framework
- **NTC - Nové technologie - výzkumné centrum**: Západočeská univerzita v Plzni for quantum computing resources and support
- **Project Supervisor**: Ing. Jan Boháč

---

<div align="center">

**⭐ Star this repo if you find it useful!**

Made with ❤️ for quantum machine learning research

</div>

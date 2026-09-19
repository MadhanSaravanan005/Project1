# Project1

Skin disease image classification pipeline utilizing quantum circuit preprocessing and convolutional neural networks on the HAM10000 dataset.

## Overview

Project1 explores hybrid quantum-classical machine learning for dermatoscopic image classification. It applies a 4-qubit parameterized Quantum Pre-processing Filter (QPF) with two CNOT entangling gates to 2x2 image patches, transforming spatial dermoscopy pixel intensities into quantum expectation states prior to classification with a Convolutional Neural Network (CNN).

## Problem

Dermoscopic skin lesion images exhibit high intra-class visual variability, subtle contrast variations, and irregular boundary morphology across different diagnostic classes (such as melanoma, melanocytic nevus, and basal cell carcinoma). Classical filtering and pooling can lose fine-grained spatial relationships. This project investigates whether mapping pixel neighborhoods into quantum state amplitudes and entangling them can provide distinct pre-processed representations for deep learning classifiers.

## Solution

The pipeline implements:
1. **Patch Extraction & Parameter Encoding**: Standardizes dermatoscopic images to 64x64 single-channel representations, dividing each image into non-overlapping 2x2 blocks. Each 4-pixel block $\phi = [\phi_1, \phi_2, \phi_3, \phi_4]$ is mapped to rotation angles $\theta_i = \pi \phi_i$.
2. **4-Qubit Quantum Circuit Simulation**:
   - Single-qubit Pauli-Y rotation gates $RY(\theta_i)$ prepare an initial state $| \psi \rangle = \bigotimes_{i=1}^4 RY(\theta_i) |0\rangle$.
   - A 16x16 unitary permutation operator $T_2$ applies two CNOT gates: $CNOT(q_1 \to q_2)$ and $CNOT(q_0 \to q_3)$.
   - Output measurement computes Pauli-$Z$ expectation values $\langle Z_i \rangle = P(q_i=0) - P(q_i=1) = 2P(q_i=0) - 1 \in [-1, 1]$.
   - Expectation values are normalized back to image intensity range $[0, 1]$ via $(\langle Z_i \rangle + 1) / 2$.
3. **Convolutional Neural Network Classification**:
   - 3 convolution/ReLU/max-pooling blocks followed by a fully connected layer and softmax output across 7 diagnostic categories.
   - Trained using RMSProp or SGDM optimizers with fixed seeds for reproducibility.

## Features

- **4-Qubit Quantum Simulation**: Statevector simulation implementing $RY(\theta)$ rotations and two CNOT entangling gates.
- **Quantum Pre-processing Filter (QPF)**: Block-wise 2x2 quantum patch transformation for 64x64 images.
- **Pre-trained Model Checkpoints**: Tracked checkpoints for both RMSProp and SGDM trained models with full training logs (1875 iterations, ~65.90% validation accuracy).
- **Graceful Dataset Fallback**: Preprocessing handles absent raw image archives cleanly and preserves pre-computed data.
- **Dual-Stack Reproducibility**: Includes canonical MATLAB implementation and a complementary pure-Python verification suite with unit tests and CLI validation.

## Tech Stack

- **Core Application**: MATLAB (Deep Learning Toolbox, Image Processing Toolbox, Parallel Computing Toolbox)
- **Verification & Testing**: Python 3.10+, NumPy, SciPy, Pandas, Pytest
- **Dataset Format**: MAT-file (v7/v5), CSV

## Project Structure

```text
Project1/
├── circuit_two_cnots.m                  # 4-qubit quantum circuit with 2 CNOTs (MATLAB)
├── preprocess_HAM10000_QPF.m            # Image preprocessing & QPF block filtering (MATLAB)
├── train_HAM10000_QPF.m                 # CNN training & validation script (MATLAB)
├── test_quantum_circuit.m               # Automated unit tests for MATLAB
├── quantum_circuit.py                   # Pure-Python implementation of quantum circuit
├── verify_pipeline.py                   # End-to-end verification CLI script
├── tests/
│   └── test_quantum_circuit.py          # Pytest suite (10 automated unit tests)
├── HAM10000_metadata.csv                # HAM10000 dataset metadata (10,015 records)
├── preprocessed_HAM10000_QPF.mat        # Preprocessed 64x64 dataset (ZTrain & label_array)
├── trained_network_HAM10000_RMSProp.mat # Trained network checkpoint (RMSProp optimizer)
├── trained_network_HAM10000_SGDM.mat    # Trained network checkpoint (SGDM optimizer)
├── requirements.txt                     # Python dependencies for testing/verification
├── .gitignore                           # Git ignore rules
└── README.md                            # Project documentation
```

## Prerequisites

- **For MATLAB execution**:
  - MATLAB R2020b or newer
  - Deep Learning Toolbox
  - Image Processing Toolbox
  - (Optional) Parallel Computing Toolbox
- **For Python verification & test suite**:
  - Python 3.10 or newer
  - pip package manager

## Installation

Clone the repository and install the Python testing dependencies:

```bash
git clone https://github.com/MadhanSaravanan005/Project1.git
cd Project1
pip install -r requirements.txt
```

## Environment Variables

No external API keys, database credentials, or secret environment variables are required. All metadata, preprocessed datasets, and trained model files are self-contained within the repository.

## Running the Project

### MATLAB Workflow

1. **Verify Quantum Circuit**:
   ```matlab
   run('test_quantum_circuit.m')
   ```

2. **Preprocess Images (Optional)**:
   If you have downloaded the raw HAM10000 image parts (`HAM10000_images_part_1` and `HAM10000_images_part_2`), run:
   ```matlab
   preprocess_HAM10000_QPF
   ```
   *Note: The preprocessed dataset is already included in `preprocessed_HAM10000_QPF.mat`.*

3. **Train the CNN Classifier**:
   ```matlab
   % Default solver: RMSProp
   train_HAM10000_QPF

   % Or train with SGDM:
   optimizer_choice = 'sgdm';
   train_HAM10000_QPF
   ```

### Python Verification Workflow

To verify the entire pipeline (metadata, quantum gates, dataset matrices, and model checkpoints) without requiring a MATLAB license:

```bash
python verify_pipeline.py
```

## Testing

Run the automated Pytest test suite:

```bash
python -m pytest tests/ -v
```

Test coverage includes:
- Orthogonality and unitarity of the 16x16 CNOT entanglement matrix ($T_2^T T_2 = I_{16}$)
- Ground state evolution ($\phi = [0, 0, 0, 0] \implies \langle Z \rangle = [1, 1, 1, 1]$)
- Entangled state evolution with target flips ($\phi = [1, 1, 1, 1] \implies \langle Z \rangle = [-1, -1, 1, 1]$)
- Expectation value bounded checks ($\langle Z_i \rangle \in [-1, 1]$)
- Block filter normalization to $[0, 1]$ range
- Dimension validation and error handling
- Dataset metadata integrity (10,015 records across 7 diagnostic categories)
- Preprocessed `.mat` array dimensions and data types
- Model checkpoint metrics and training info verification

## Example Usage

### Simulating the Quantum Circuit in Python

```python
import numpy as np
from quantum_circuit import circuit_two_cnots, apply_qpf_block

# 4 normalized pixel values from a 2x2 image patch
pixels = [0.1, 0.4, 0.6, 0.9]
expvals = circuit_two_cnots(pixels)
print("Pauli-Z Expectation Values:", expvals)

# Applying filter to a 2x2 image patch
patch = np.array([[0.2, 0.5], [0.7, 0.9]])
filtered_patch = apply_qpf_block(patch)
print("Filtered Patch:\n", filtered_patch)
```

### Calling the Quantum Circuit in MATLAB

```matlab
phi = [0.1; 0.4; 0.6; 0.9];
expval = circuit_two_cnots(phi);
disp('Pauli-Z Expectation values:');
disp(expval);
```

## Limitations

- **Severe Class Imbalance**: The HAM10000 dataset is heavily unbalanced. Of the 10,015 samples, 6,705 (~67%) belong to `nevus`, while `df` has only 115 samples (~1.1%). As recorded in the trained models, models trained without class weighting achieve ~65.90% validation accuracy, primarily learning to predict the majority class.
- **Raw Image Storage**: The 10,015 raw full-resolution images (~2.6 GB) are not committed to Git to keep repository clone times practical; instead, the preprocessed 64x64 dataset (`preprocessed_HAM10000_QPF.mat`, 28 MB) is tracked for immediate training.
- **MATLAB Deep Learning License**: End-to-end retraining with `trainNetwork` requires a proprietary MATLAB license with the Deep Learning Toolbox.

## Future Improvements

- Incorporate focal loss or weighted cross-entropy loss to improve sensitivity on under-represented classes (e.g. melanoma, dermatofibroma).
- Implement an end-to-end PyTorch or PennyLane equivalent for native open-source retraining and GPU acceleration without MATLAB.
- Test parameterized entangling gates (e.g., parameterized $CR_X$ / $CR_Z$) rather than fixed CNOT permutations.

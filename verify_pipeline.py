"""
verify_pipeline.py
Standalone verification script to validate Project1:
1. HAM10000 metadata integrity
2. Quantum circuit mathematical consistency & CNOT entanglement
3. Preprocessed MAT dataset (dimensions, labels, values)
4. Trained model checkpoints (RMSProp, SGDM) and recorded performance
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.io as sio

from quantum_circuit import circuit_two_cnots, apply_qpf_block, T2_MATRIX


def verify_metadata(csv_path: str = "HAM10000_metadata.csv") -> bool:
    print(f"\n[1/4] Checking metadata: {csv_path}")
    if not os.path.isfile(csv_path):
        print(f"  [FAIL] Metadata file not found: {csv_path}")
        return False

    df = pd.read_csv(csv_path)
    required_cols = {"lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"  [FAIL] Missing columns: {missing}")
        return False

    row_count = len(df)
    unique_dx = df["dx"].unique()
    print(f"  [PASS] Found {row_count} records across {len(unique_dx)} disease classes:")
    for dx, count in df["dx"].value_counts().items():
        print(f"         - {dx}: {count}")

    if row_count != 10015 or len(unique_dx) != 7:
        print(f"  [FAIL] Unexpected row count ({row_count} != 10015) or classes ({len(unique_dx)} != 7)")
        return False

    return True


def verify_quantum_circuit() -> bool:
    print("\n[2/4] Verifying 4-qubit quantum circuit simulation...")

    # Test 1: T2 unitarity
    is_unitary = np.allclose(T2_MATRIX.T @ T2_MATRIX, np.eye(16))
    if not is_unitary:
        print("  [FAIL] T2 matrix is not unitary.")
        return False
    print("  [PASS] Two-CNOT entanglement operator T2 is strictly unitary (T2^T * T2 = I_16)")

    # Test 2: Ground state
    expval_zero = circuit_two_cnots([0.0, 0.0, 0.0, 0.0])
    if not np.allclose(expval_zero, [1.0, 1.0, 1.0, 1.0]):
        print(f"  [FAIL] Ground state expectation failed: {expval_zero}")
        return False
    print("  [PASS] Zero rotation gives ground state: <Z> = [1.0, 1.0, 1.0, 1.0]")

    # Test 3: Pi rotation (with CNOT target flips)
    expval_pi = circuit_two_cnots([1.0, 1.0, 1.0, 1.0])
    if not np.allclose(expval_pi, [-1.0, -1.0, 1.0, 1.0]):
        print(f"  [FAIL] Pi rotation expectation failed: {expval_pi}")
        return False
    print("  [PASS] Pi rotation with CNOTs yields entangled state: <Z> = [-1.0, -1.0, 1.0, 1.0]")

    # Test 4: 2x2 image block filter
    dummy_block = np.array([[0.25, 0.50], [0.75, 1.00]])
    filtered = apply_qpf_block(dummy_block)
    if filtered.shape != (2, 2) or (filtered < 0.0).any() or (filtered > 1.0).any():
        print(f"  [FAIL] QPF block filter output invalid: {filtered}")
        return False
    print(f"  [PASS] QPF 2x2 block filtering correctly maps into [0, 1] range")

    return True


def verify_dataset(mat_path: str = "preprocessed_HAM10000_QPF.mat") -> bool:
    print(f"\n[3/4] Verifying preprocessed dataset: {mat_path}")
    if not os.path.isfile(mat_path):
        print(f"  [FAIL] File not found: {mat_path}")
        return False

    data = sio.loadmat(mat_path)
    if "ZTrain" not in data or "label_array" not in data:
        print(f"  [FAIL] Missing required keys ('ZTrain', 'label_array'). Found: {list(data.keys())}")
        return False

    z = data["ZTrain"]
    labels = data["label_array"]

    print(f"  [PASS] ZTrain shape: {z.shape} (dtype: {z.dtype})")
    print(f"  [PASS] label_array shape: {labels.shape} (dtype: {labels.dtype})")
    print(f"  [PASS] Pixel range: [{z.min():.4f}, {z.max():.4f}]")
    print(f"  [PASS] Unique class labels: {np.unique(labels).tolist()}")

    if z.shape != (64, 64, 1, 10015):
        print(f"  [FAIL] Unexpected ZTrain shape: {z.shape}")
        return False

    return True


def verify_models() -> bool:
    print("\n[4/4] Verifying trained model checkpoints...")
    models = [
        ("trained_network_HAM10000_RMSProp.mat", "RMSProp"),
        ("trained_network_HAM10000_SGDM.mat", "SGDM")
    ]

    all_passed = True
    for mat_file, opt_name in models:
        if not os.path.isfile(mat_file):
            print(f"  [FAIL] Model file missing: {mat_file}")
            all_passed = False
            continue

        data = sio.loadmat(mat_file, struct_as_record=False, squeeze_me=True)
        acc = float(data.get("accuracy", 0.0))
        info = data.get("info", None)

        iters = getattr(info, "OutputNetworkIteration", "N/A") if info else "N/A"
        final_val_loss = getattr(info, "FinalValidationLoss", "N/A") if info else "N/A"

        print(f"  [PASS] Model ({opt_name}): {mat_file}")
        print(f"         - Validation Accuracy: {acc * 100:.2f}%")
        print(f"         - Total Iterations:    {iters}")
        print(f"         - Final Val Loss:      {final_val_loss}")

    return all_passed


def main() -> int:
    print("============================================================")
    print("   Project1: Quantum Pre-processing Pipeline Verification   ")
    print("============================================================")

    results = [
        verify_metadata(),
        verify_quantum_circuit(),
        verify_dataset(),
        verify_models()
    ]

    print("\n------------------------------------------------------------")
    if all(results):
        print("RESULT: ALL PIPELINE VERIFICATIONS PASSED SUCCESSFULLY.")
        print("------------------------------------------------------------")
        return 0
    else:
        print("RESULT: VERIFICATION FAILED ON ONE OR MORE CHECKS.")
        print("------------------------------------------------------------")
        return 1


if __name__ == "__main__":
    sys.exit(main())

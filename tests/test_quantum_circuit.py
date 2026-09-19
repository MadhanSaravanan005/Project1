"""
Unit tests for Project1 quantum pre-processing pipeline and data integrity.
"""

import os
import pytest
import numpy as np
import pandas as pd
import scipy.io as sio

from quantum_circuit import circuit_two_cnots, apply_qpf_block, T2_MATRIX


def test_unitary_two_cnot_matrix():
    """Verify that the 16x16 entanglement matrix T2 is orthogonal/unitary."""
    product = T2_MATRIX.T @ T2_MATRIX
    identity = np.eye(16)
    assert np.allclose(product, identity, atol=1e-10)


def test_ground_state_all_zeros():
    """When angles are 0, state remains |0000>, so all qubits yield <Z> = +1."""
    expval = circuit_two_cnots([0.0, 0.0, 0.0, 0.0])
    assert expval.shape == (4,)
    assert np.allclose(expval, [1.0, 1.0, 1.0, 1.0], atol=1e-7)


def test_pi_rotation_with_entanglement():
    """When angles are 1 (pi rotation), state |1111> evolves under T2 into |1100>."""
    expval = circuit_two_cnots([1.0, 1.0, 1.0, 1.0])
    assert expval.shape == (4,)
    assert np.allclose(expval, [-1.0, -1.0, 1.0, 1.0], atol=1e-7)


def test_arbitrary_inputs_within_bounds():
    """Verify that for arbitrary inputs in [0, 1], Pauli-Z expectation values stay within [-1, 1]."""
    np.random.seed(42)
    for _ in range(50):
        phi = np.random.uniform(0.0, 1.0, size=4)
        expval = circuit_two_cnots(phi)
        assert np.all(expval >= -1.0 - 1e-7)
        assert np.all(expval <= 1.0 + 1e-7)


def test_invalid_input_dimensions():
    """Ensure circuit_two_cnots raises ValueError when input length is not 4."""
    with pytest.raises(ValueError):
        circuit_two_cnots([0.5, 0.5])

    with pytest.raises(ValueError):
        circuit_two_cnots([0.1, 0.2, 0.3, 0.4, 0.5])


def test_apply_qpf_block_2x2():
    """Verify 2x2 image block filtering output dimensions and range [0, 1]."""
    block = np.array([[0.1, 0.3], [0.5, 0.7]])
    filtered = apply_qpf_block(block)
    assert filtered.shape == (2, 2)
    assert np.all(filtered >= 0.0)
    assert np.all(filtered <= 1.0)


def test_apply_qpf_block_invalid_shape():
    """Verify apply_qpf_block rejects inputs not of shape (2, 2)."""
    with pytest.raises(ValueError):
        apply_qpf_block(np.zeros((3, 3)))


def test_metadata_csv_structure():
    """Verify integrity of HAM10000_metadata.csv."""
    csv_file = "HAM10000_metadata.csv"
    assert os.path.isfile(csv_file), f"Missing {csv_file}"
    df = pd.read_csv(csv_file)
    assert len(df) == 10015
    assert "image_id" in df.columns
    assert "dx" in df.columns
    assert df["dx"].nunique() == 7
    assert df["image_id"].isnull().sum() == 0


def test_preprocessed_dataset_file():
    """Verify integrity of preprocessed_HAM10000_QPF.mat."""
    mat_file = "preprocessed_HAM10000_QPF.mat"
    assert os.path.isfile(mat_file), f"Missing {mat_file}"
    data = sio.loadmat(mat_file)
    assert "ZTrain" in data
    assert "label_array" in data
    z = data["ZTrain"]
    labels = data["label_array"]
    assert z.shape == (64, 64, 1, 10015)
    assert labels.shape == (10015, 1)
    assert z.min() >= 0.0
    assert z.max() <= 1.0


def test_trained_model_artifacts():
    """Verify trained network MAT files exist and have valid accuracy/info."""
    for model_path in ["trained_network_HAM10000_RMSProp.mat", "trained_network_HAM10000_SGDM.mat"]:
        assert os.path.isfile(model_path), f"Missing {model_path}"
        data = sio.loadmat(model_path, struct_as_record=False, squeeze_me=True)
        assert "accuracy" in data
        acc = float(data["accuracy"])
        assert 0.0 <= acc <= 1.0
        assert "info" in data

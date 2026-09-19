"""
quantum_circuit.py
Faithful Python implementation of the 4-qubit quantum circuit defined in circuit_two_cnots.m.
Simulates parameter rotations and two CNOT entangling gates, returning Pauli-Z expectation values.
"""

from __future__ import annotations
import numpy as np


def ry_gate(theta: float) -> np.ndarray:
    """Single-qubit RY rotation matrix."""
    half_theta = theta / 2.0
    return np.array([
        [np.cos(half_theta), -np.sin(half_theta)],
        [np.sin(half_theta),  np.cos(half_theta)]
    ], dtype=np.float64)


def get_two_cnot_matrix() -> np.ndarray:
    """
    Construct the 16x16 unitary permutation matrix T2 implementing:
    - CNOT with control qubit 1, target qubit 2
    - CNOT with control qubit 0, target qubit 3
    """
    mapping = [0, 1, 2, 3, 6, 7, 4, 5, 9, 8, 11, 10, 15, 14, 13, 12]
    T2 = np.zeros((16, 16), dtype=np.float64)
    for row, col in enumerate(mapping):
        T2[row, col] = 1.0
    return T2


# Precompute T2 unitary matrix
T2_MATRIX = get_two_cnot_matrix()

# Index masks where each qubit is in state |0> (0-indexed)
INDEX_Q0_ZERO = np.arange(0, 8)
INDEX_Q1_ZERO = np.concatenate([np.arange(0, 4), np.arange(8, 12)])
INDEX_Q2_ZERO = np.array([0, 1, 4, 5, 8, 9, 12, 13])
INDEX_Q3_ZERO = np.arange(0, 16, 2)


def circuit_two_cnots(phi: np.ndarray | list[float]) -> np.ndarray:
    """
    Simulate the 4-qubit quantum circuit.

    Args:
        phi: 4-element numeric array representing normalized pixel values [phi1, phi2, phi3, phi4].

    Returns:
        np.ndarray: 1D array of 4 Pauli-Z expectation values in range [-1.0, 1.0].
    """
    phi = np.asarray(phi, dtype=np.float64).flatten()
    if phi.size != 4:
        raise ValueError(f"Input phi must have exactly 4 elements, got {phi.size}.")

    angles = np.pi * phi

    # Compute single-qubit rotations
    ry1 = ry_gate(angles[0])
    ry2 = ry_gate(angles[1])
    ry3 = ry_gate(angles[2])
    ry4 = ry_gate(angles[3])

    # Separable state operator T1 = RY1 (x) RY2 (x) RY3 (x) RY4
    T1 = np.kron(ry1, np.kron(ry2, np.kron(ry3, ry4)))

    # Ground state |0000>
    x = np.zeros(16, dtype=np.float64)
    x[0] = 1.0

    # State evolution through rotation and two CNOTs
    y = T2_MATRIX @ (T1 @ x)

    # Probability of measuring |0> for each qubit
    p0 = np.array([
        np.sum(np.abs(y[INDEX_Q0_ZERO]) ** 2),
        np.sum(np.abs(y[INDEX_Q1_ZERO]) ** 2),
        np.sum(np.abs(y[INDEX_Q2_ZERO]) ** 2),
        np.sum(np.abs(y[INDEX_Q3_ZERO]) ** 2)
    ], dtype=np.float64)

    # Pauli-Z expectation value: <Z> = P(0) - P(1) = 2*P(0) - 1
    expval_all = 2.0 * p0 - 1.0
    return expval_all


def apply_qpf_block(block: np.ndarray) -> np.ndarray:
    """
    Apply Quantum Pre-processing Filter to a 2x2 image block.
    Maps expectation values [-1, 1] back to normalized image intensity [0, 1].
    """
    if block.shape != (2, 2):
        raise ValueError(f"Expected (2, 2) block, got {block.shape}.")

    expvals = circuit_two_cnots(block.flatten())
    # <Z> = 2*P(0) - 1 => P(0) = (<Z> + 1) / 2
    normalized_block = (expvals.reshape((2, 2)) + 1.0) / 2.0
    return normalized_block

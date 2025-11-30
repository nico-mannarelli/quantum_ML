"""
Quantum Angle Encoding Example

This module demonstrates how to encode classical data into quantum states
using rotation gates (RX and RY gates) in Qiskit.
"""

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

def angle_encoding_example():
    """
    Demonstrates angle encoding: encoding classical data using rotation gates.
    
    Returns:
        Statevector: The quantum state after encoding
    """
    # Create a quantum circuit with 2 qubits
    qc = QuantumCircuit(2)
    
    # Encode classical data using rotation gates
    angle1 = 0.3  # Example angle for encoding qubit 0
    angle2 = 0.6  # Example angle for encoding qubit 1
    
    qc.rx(angle1, 0)  # Rotate qubit 0 around X-axis
    qc.ry(angle2, 1)  # Rotate qubit 1 around Y-axis
    
    # Simulate quantum circuit and get statevector directly
    statevector = Statevector.from_instruction(qc)
    
    print("Angle Encoding Example")
    print("=" * 50)
    print(f"Angle 1 (RX on qubit 0): {angle1}")
    print(f"Angle 2 (RY on qubit 1): {angle2}")
    print("\nResulting Statevector:")
    print(statevector)
    print("\nCircuit:")
    print(qc)
    
    return statevector

if __name__ == "__main__":
    angle_encoding_example()


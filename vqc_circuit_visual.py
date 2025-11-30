"""
Variational Quantum Circuit (VQC) Visualization

This module demonstrates how to create and visualize a variational quantum circuit
using PennyLane. The circuit uses angle embedding and strongly entangling layers
for quantum machine learning applications.
"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt


def create_vqc_circuit(n_qubits=2, n_layers=2):
    """
    Create a Variational Quantum Circuit with angle embedding and entangling layers.
    
    Args:
        n_qubits (int): Number of qubits in the circuit
        n_layers (int): Number of strongly entangling layers
        
    Returns:
        tuple: (quantum device, quantum node, circuit function)
    """
    # Quantum device
    dev = qml.device("default.qubit", wires=n_qubits)

    # Quantum circuit definition
    def circuit(params, state):
        """
        Variational quantum circuit with angle embedding and entangling layers.
        
        Args:
            params: Parameters for the strongly entangling layers
            state: Input classical data to encode into quantum state
            
        Returns:
            list: Expectation values of Pauli-Z operators
        """
        qml.AngleEmbedding(state, wires=range(n_qubits), rotation='Y')
        qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    # Create quantum node
    @qml.qnode(dev)
    def q_network(params, state):
        return circuit(params, state)

    return dev, q_network, circuit


def visualize_circuit(n_qubits=2, n_layers=2, output_file="circuit_diagram.txt"):
    """
    Generate and visualize a variational quantum circuit.
    
    Args:
        n_qubits (int): Number of qubits
        n_layers (int): Number of layers
        output_file (str): File to save the circuit diagram
    """
    # Initialize random parameters
    params = 0.1 * np.random.randn(n_layers, n_qubits, 3)
    
    # Create circuit
    dev, q_network, circuit = create_vqc_circuit(n_qubits, n_layers)
    
    # Example state for visualization
    state = np.random.randn(n_qubits)
    
    # Draw the circuit
    drawer = qml.draw(q_network)
    circuit_diagram = drawer(params, state)
    
    print("Variational Quantum Circuit")
    print("=" * 50)
    print(f"Number of qubits: {n_qubits}")
    print(f"Number of layers: {n_layers}")
    print("\nCircuit Diagram:")
    print(circuit_diagram)
    
    # Save to file
    with open(output_file, "w") as f:
        f.write(circuit_diagram)
    
    print(f"\nCircuit diagram saved to {output_file}")
    
    # Execute the circuit
    result = q_network(params, state)
    print(f"\nExpectation values: {result}")
    
    return circuit_diagram, result


if __name__ == "__main__":
    visualize_circuit(n_qubits=2, n_layers=2)

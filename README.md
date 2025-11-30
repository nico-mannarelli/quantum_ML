# Quantum Machine Learning Portfolio

A collection of quantum machine learning and reinforcement learning projects demonstrating classical and quantum computing techniques for machine learning applications.

## Overview

This repository showcases three key projects:

1. **Quantum Angle Encoding** - Classical data encoding into quantum states using rotation gates
2. **Deep Q-Learning (DQN)** - Reinforcement learning agent for CartPole-v1 using PyTorch
3. **Variational Quantum Circuit (VQC)** - Quantum circuit visualization and execution using PennyLane

## Project Structure

```
quant-ml-fire/
├── angle_encoding.py       # Quantum angle encoding example using Qiskit
├── deep_q_learning.py      # Deep Q-Network implementation for CartPole
├── vqc_circuit_visual.py   # Variational Quantum Circuit visualization
├── circuit_diagram.txt     # Generated circuit diagram output
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd quant-ml-fire
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Projects

### 1. Quantum Angle Encoding

Demonstrates how to encode classical data into quantum states using rotation gates (RX and RY) in Qiskit.

**Features:**
- Quantum state preparation using rotation gates
- Statevector simulation
- Circuit visualization

**Run:**
```bash
python angle_encoding.py
```

**Output:**
- Displays the encoded quantum statevector
- Shows the quantum circuit diagram

---

### 2. Deep Q-Learning (DQN)

Implementation of a Deep Q-Network agent that learns to balance a pole on a cart using reinforcement learning.

**Features:**
- Experience replay buffer
- Target network for stable training
- Epsilon-greedy exploration strategy
- PyTorch-based neural network

**Run:**
```bash
python deep_q_learning.py
```

**Training:**
- Trains for 1000 episodes on CartPole-v1 environment
- Uses epsilon-greedy exploration with decay
- Updates target network every 10 episodes

**Evaluation:**
- Evaluates the trained agent over 10 episodes
- Reports average reward achieved

---

### 3. Variational Quantum Circuit (VQC)

Creates and visualizes a variational quantum circuit using PennyLane with angle embedding and strongly entangling layers.

**Features:**
- Angle embedding for data encoding
- Strongly entangling layers for quantum feature maps
- Expectation value measurements (Pauli-Z)
- Circuit diagram generation

**Run:**
```bash
python vqc_circuit_visual.py
```

**Output:**
- Prints the circuit diagram to console
- Saves circuit diagram to `circuit_diagram.txt`
- Displays expectation values

## Technical Details

### Quantum Angle Encoding
- **Framework:** Qiskit
- **Simulator:** AerSimulator
- **Encoding Method:** Rotation gates (RX, RY)

### Deep Q-Learning
- **Framework:** PyTorch
- **Environment:** OpenAI Gym (CartPole-v1)
- **Algorithm:** DQN with experience replay and target network
- **Network Architecture:** 3-layer fully connected (128-128-output)

### Variational Quantum Circuit
- **Framework:** PennyLane
- **Device:** Default qubit simulator
- **Encoding:** Angle embedding (Y-rotation)
- **Ansatz:** Strongly entangling layers

## Dependencies

### Core Libraries
- **qiskit** - Quantum computing framework by IBM
- **qiskit-aer** - Quantum circuit simulators
- **pennylane** - Quantum machine learning framework
- **torch** - PyTorch for deep learning
- **gym** - OpenAI Gym for reinforcement learning environments
- **numpy** - Numerical computing
- **scipy** - Scientific computing
- **matplotlib** - Visualization

## Key Learning Outcomes

This portfolio demonstrates:
- **Quantum Computing:** Understanding of quantum circuits, state preparation, and quantum simulators
- **Machine Learning:** Implementation of deep reinforcement learning algorithms
- **Hybrid Systems:** Combining classical and quantum computing for ML applications
- **Software Engineering:** Clean code structure, documentation, and project organization

## Notes

- The DQN implementation includes gradient clipping for training stability
- All quantum circuits use simulators (no quantum hardware required)
- Circuit diagrams are generated in text format for easy viewing


## License

This project is open source and available for educational purposes.

---

**Author:** Nico Mannarelli  
**Date:** 2024


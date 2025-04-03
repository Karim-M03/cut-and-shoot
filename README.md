# My Prototype of Cut&Shoot

This repository presents a prototype developed as part of my thesis project. This repo offers an implementation of the **Cut & Shoot pipeline** a hybrid quantum-classical approach that combines quantum circuit cutting and shot-wise distribution to evaluate large quantum circuits using multiple NISQ devices/simulators.

  This pipeline is designed to solve current hardware limitations by distributing the workloads across multiple backends and reconstructing the output of each via classical post-processing.

This code builds on from the following papers:

- **Cut & Shoot: Cutting & Distributing Quantum Circuits Across Multiple NISQ Computers**  
  *Giuseppe Bisicchia, Alessandro Bocci, José García-Alonso, Juan M. Murillo, Antonio Brogi*

  See `cut_and_shoot.pdf`

- **CutQC: Using Small Quantum Computers for Large Quantum Circuit Evaluations**  
  *Wei Tang, Teague Tomesh, Martin Suchara, Jeffrey Larson, Margaret Martonosi*

  See `cutqc.pdf`

---

## Table of Contents

- [Key Features](#key-features)  
- [Repository Structure](#repository-structure)  
- [Installation & Setup](#installation--setup)  
- [Usage](#usage)  
- [Code Walkthrough](#code-walkthrough)  
- [References](#references)  
- [License](#license)

---

## Key Features

### Circuit Cutting
- Automatically extracts subcircuits from a large circuit
- Generates all required input/output basis variants for measurement stitching

### Shot-Wise Distribution
- Splits shots across QPUs
- TODO: define users policies for excluding/prioritizing QPUs

### Classical Post-Processing
- Combines partial results using two reconstruction methods:
  - **Full Distribution (FD)** – Enumerates all combinations
  - **Dynamic Definition (DD)** – Reconstructs entry-by-entr

### Extensible
- Easily integrate new QPU backends
- Modify cut strategies, solvers, or distribution policies

---

## Repository Structure

```
.
├── examples/               # sample circuits
│   ├── grover.py
│   ├── rca.py
│   └── ...
|    
├── constructor.py         # vuilds subcircuits and variants
├── graph.py               # graph extraction
├── main.py                # main entry point for running pipeline
├── merge.py               # merges measurement results
├── model.py               # MILP model for subcircuit & shot assignment
├── qpu.py                 # models QPU properties
├── runner.py              # dispatches circuits to QPUs
└── load_credentials.py    # loads IBM Quantum account credential
```

---

## Installation & Setup

1. **Clone the Repository**
```bash
git clone https://github.com/Karim-M03/cut-and-shoot.git
cd cut-and-shoot
```

2. **Create a Virtual Environment (highly recommended)**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **(Optional) Configure IBM Q**
- Update `load_credentials.py` with your token
- run it
---

## Usage

Edit `main.py` to select your desired test circuit and qpus:

```python
from examples import simple_circuit, rca, grover

qc = simple_circuit.simple_circuit()
# or: qc = rca.ripple_carry_adder(5)
# or: qc = grover.grover_circuit(2)
```

```python
def create_qpus():
    # QPU names to create
    qpu_types = [
        'aer_simulator',
        'aer_simulator_statevector',
        'ibmq_qasm_simulator',
        'ibm_nairobi',
        'ibm_oslo',
    ]

    # Optional override: (qpu_type, execution_time, queue_time, capacity)
    qpu_override_params = {
        'aer_simulator': (10, 1, 10),
        'aer_simulator_statevector': (12, 2, 8),
        'ibmq_qasm_simulator': (60, 3, 6),
        'ibm_nairobi': (300, 6, 2),
        'ibm_oslo': (280, 5, 3),
    }

    qpus = []
    for i, qpu_type in enumerate(qpu_types):
        qpu = QPU(qpu_type, i)
        # comment this if you don't want to override QPU's metrics
         """if qpu_type in qpu_override_params:
            exec_time, queue_time, capacity = qpu_override_params[qpu_type]
            qpu.update_metrics(exec_time, queue_time, capacity) """

        qpus.append(qpu)

    return qpus
```

Then run:

```bash
python main.py
```

This will:
- Convert the circuit to a DAG and extract graph data
- Solve the MILP model to divide the circuit into multiple subcircuits, assigning them shots to specific QPUs
- Run subcircuits with the assigned shots
- Merge results and reconstruct the final output

---

## Code Walkthrough

- **`model.py`**  
  Defines the `CutAndShootModel` using PuLP, which chooses how to cut the circuit and distribute shots optimally

- **`constructor.py`**  
  Converts model output into subcircuits, generating all required input/output basis variants

- **`runner.py`**  
  Handles QPU execution of subcircuits in parallel

- **`merge.py` **  
  Combines results
- missing final reconstruction

---

## References

- **Cut & Shoot:**  
  Bisicchia et al. (2023), *Cut & Shoot: Cutting & Distributing Quantum Circuits Across Multiple NISQ Computers*

- **CutQC:**  
  Tang et al. (2021), *Using Small Quantum Computers for Large Quantum Circuit Evaluations*, ASPLOS

---

#!/usr/bin/env python3

from qiskit import QuantumCircuit
import numpy as np


def oracle_circuit(num_qubits):
    # oracle that marks (phase flip) the state |0...0>
    # apply X on all qubits
    # put the last qubit in the |+> basis
    # apply multi-controlled X with all but the last as controls
    # return the last qubit to computational basis
    # apply X again to all qubits

    oracle = QuantumCircuit(num_qubits)

    oracle.x(range(num_qubits))
    oracle.h(num_qubits - 1)
    oracle.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    oracle.h(num_qubits - 1)
    oracle.x(range(num_qubits))

    return oracle

def diffuser_circuit(num_qubits):
    # standard diffuser (inversion about the average)
    # apply H, X, H, MCX, H, X, H in the proper sequence

    diffuser = QuantumCircuit(num_qubits)

    diffuser.h(range(num_qubits))
    diffuser.x(range(num_qubits))
    diffuser.h(num_qubits - 1)
    diffuser.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    diffuser.h(num_qubits - 1)
    diffuser.x(range(num_qubits))
    diffuser.h(range(num_qubits))

    return diffuser

def grover_circuit(num_qubits):

    qc = QuantumCircuit(num_qubits, num_qubits)

    qc.h(range(num_qubits))

    # number of iterations: ~ pi/4 * sqrt(2^n)
    num_iterations = int(np.round(np.pi / 4 * np.sqrt(2 ** num_qubits)))

    # convert oracle and diffuser to gates for reuse
    oracle_gate = oracle_circuit(num_qubits).to_gate(label='oracle')
    diffuser_gate = diffuser_circuit(num_qubits).to_gate(label='diffuser')

    for _ in range(num_iterations):
        qc.append(oracle_gate, range(num_qubits))
        qc.append(diffuser_gate, range(num_qubits))

    qc.measure(range(num_qubits), range(num_qubits))

    return qc



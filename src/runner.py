from collections import defaultdict
from qiskit import transpile, ClassicalRegister, QuantumCircuit
from qiskit.providers import Backend
from typing import List, Tuple
from copy import deepcopy
import logging
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

from subcircuit import Variant

# suppress Qiskit logs
""" for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.CRITICAL)
logging.getLogger('qiskit').setLevel(logging.CRITICAL)
 """



def run_circuit_list(
    circuit_list: List[Variant],
    backend: Backend
) -> dict[str, dict[str, dict]]:
    """
    executes a list of quantum circuits with associated metadata.

    :param circuit_list: list of Variant
    :param backend: Qiskit backend
    :return: dict mapping circuit name to execution results and metadata
    """
    results = {}

    grouped = defaultdict(list)
    variant_map = {}


    for variant in circuit_list:
        qc = variant.circuit
        if qc.num_qubits == 0:
            continue

        qc_copy = deepcopy(qc)

        if qc_copy.count_ops().get("measure", 0) == 0:
            creg = ClassicalRegister(qc_copy.num_qubits, f"auto_meas_{variant.name}")
            qc_copy.add_register(creg)
            for i in range(qc_copy.num_qubits):
                qc_copy.measure(i, i)

        grouped[variant.shots].append((variant.name, qc_copy))
        variant_map[variant.name] = variant

    for shots, group in grouped.items():
        names, qcs = zip(*group)
        transpiled = transpile(list(qcs), backend)
        job = backend.run(transpiled, shots=shots)
        result = job.result()

        for i, name in enumerate(names):
            counts = result.get_counts(i)
            total_shots = sum(counts.values())
            results[name] = {
                'counts': dict(counts),
                'total_shots': total_shots
            }

    return results

from collections import defaultdict
from qiskit import transpile, ClassicalRegister, QuantumCircuit
from qiskit.providers import Backend
from typing import List, Tuple
from copy import deepcopy
import logging
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# suppress Qiskit logs
""" for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.CRITICAL)
logging.getLogger('qiskit').setLevel(logging.CRITICAL)
 """



def run_circuit_list(
    circuit_list: List[Tuple[str, QuantumCircuit, int, List[int], List[int], List[int]]],
    backend: Backend
) -> dict[str, dict[str, dict]]:
    """
    executes a list of quantum circuits with associated metadata.

    :param circuit_list: list of tuples (circuit_name, QuantumCircuit, shots, active_qubits, init_qbits, meas_qbits)
    :param backend: Qiskit backend
    :return: dict mapping circuit name to execution results and metadata
    """
    results = {}

    grouped = defaultdict(list)
    active_qubits_map = {}
    init_qbits_map = {}
    meas_qbits_map = {}

    for name, qc, shots, active_qubits, init_qbits, meas_qbits in circuit_list:
        if qc.num_qubits == 0:
            continue

        qc_copy = deepcopy(qc)

        if qc_copy.count_ops().get("measure", 0) == 0:
            creg = ClassicalRegister(qc_copy.num_qubits, f"auto_meas_{name}")
            qc_copy.add_register(creg)
            for i in range(qc_copy.num_qubits):
                qc_copy.measure(i, i)

        grouped[shots].append((name, qc_copy))
        active_qubits_map[name] = active_qubits
        init_qbits_map[name] = init_qbits
        meas_qbits_map[name] = meas_qbits

    for shots, group in grouped.items():
        names, qcs = zip(*group)
        transpiled = transpile(list(qcs), backend)
        job = backend.run(transpiled, shots=shots)
        result = job.result()

        for i, name in enumerate(names):
            counts = result.get_counts(i)
            total_shots = sum(counts.values())
            probabilities = {k: v / total_shots for k, v in counts.items()}

            results[name] = {
                'counts': dict(counts),
                'probabilities': probabilities,
                'active_qubits': active_qubits_map.get(name, []),
                'initialized_qubits': init_qbits_map.get(name, []),
                'measured_qubits': meas_qbits_map.get(name, []),
                'total_shots': total_shots
            }

    return results

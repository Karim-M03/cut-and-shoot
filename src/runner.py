from collections import defaultdict
from qiskit import transpile, ClassicalRegister, QuantumCircuit
from qiskit.providers import Backend
from typing import List, Tuple
from copy import deepcopy
import logging

# suppress Qiskit logs
""" for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.CRITICAL)
logging.getLogger('qiskit').setLevel(logging.CRITICAL)
 """



def run_circuit_list(
    circuit_list: List[Tuple[str, QuantumCircuit, int, List[int]]],
    backend: Backend
) -> dict[str, dict[str, dict]]:
    """
    executes a list of quantum circuits with associated active qubits.

    :param circuit_list: list of (circuit_name, QuantumCircuit, shots, active_qubits)
    :param backend: Backend instance
    :return: dict {circuit_name: {'counts': {...}, 'active_qubits': [...]}, ...}
    """
    results = {}

    # group circuits by shot count
    grouped = defaultdict(list)
    active_qubits_map = {}

    for name, qc, shots, active_qubits in circuit_list:
        if qc.num_qubits == 0:
            continue

        # deepcopy to isolate circuits
        qc_copy = deepcopy(qc)

        # add default measurement if needed
        if qc_copy.count_ops().get("measure", 0) == 0:
            creg = ClassicalRegister(qc_copy.num_qubits, f"auto_meas_{name}")
            qc_copy.add_register(creg)
            for i in range(qc_copy.num_qubits):
                qc_copy.measure(i, i)

        grouped[shots].append((name, qc_copy))
        active_qubits_map[name] = active_qubits

    # execute circuits by group
    for shots, group in grouped.items():
        names, qcs = zip(*group)
        transpiled = transpile(list(qcs), backend)
        job = backend.run(transpiled, shots=shots)
        result = job.result()

        for i, name in enumerate(names):
            counts = result.get_counts(i)
            print(f"Executed: {name} → {counts}")
            results[name] = {
                'counts': dict(counts),
                'active_qubits': active_qubits_map.get(name, [])
            }

    return results

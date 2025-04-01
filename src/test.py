import json
from pprint import pprint
from typing import List, Optional, Tuple

import networkx as nx
import pennylane as qml
from pennylane import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.converters import circuit_to_dag

from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

from constructor import create_quantum_subcircuits
from graph import get_dag_mapping, extract_graph_data
from merge import merge_and_normalize_variant_counts
from model import CutAndShootModel
from post_processing import fd_reconstruct_variant_dict, dd_reconstruct_variant_dict
import post_processing_2
from qpu import QPU
from runner import run_circuit_list


def create_qpus():
    """Create a list of QPUs and optionally override their metrics."""
    qpu_types = [
        'aer_simulator',
        'aer_simulator_statevector',
        'ibmq_qasm_simulator',
        'ibm_nairobi',
        'ibm_oslo',
    ]

    qpu_override_params = {
        'aer_simulator': (10, 1, 200),
        'aer_simulator_statevector': (12, 2, 200),
        'ibmq_qasm_simulator': (60, 3, 200),
        'ibm_nairobi': (300, 6, 200),
        'ibm_oslo': (280, 5, 300),
    }

    qpus = []
    for i, qpu_type in enumerate(qpu_types):
        qpu = QPU(qpu_type, i)
        if qpu_type in qpu_override_params:
            exec_time, queue_time, capacity = qpu_override_params[qpu_type]
            qpu.update_metrics(exec_time, queue_time, capacity)
        qpus.append(qpu)

    return qpus


def test(qc):
    dag = circuit_to_dag(qc)
    id_mapping = get_dag_mapping(dag)
    vertex_weights, edges = extract_graph_data(dag, id_mapping)

    qpus = create_qpus()

    model = CutAndShootModel(
        edges=edges,
        vertex_weights=vertex_weights,
        qpus=qpus,
        num_shots_per_subcircuit=1024,
        num_subcircuits=4,
        alpha=0.8,
        beta=0.2
    )

    model.solve_model()
    subcircuits, num_cuts = model.print_and_return_solution()
    quantum_subcircuits = create_quantum_subcircuits(subcircuits, qc, id_mapping=id_mapping)

    qpu_assignments = {qpu.index: [] for qpu in qpus}
    for subcircuit_id, variants in quantum_subcircuits.items():
        for qpu_index, shots in subcircuits[subcircuit_id]['shots'].items():
            if shots > 0:
                for name, (circuit, active_qubits) in variants.items():
                    qpu_assignments[qpu_index].append((name, circuit, shots, active_qubits))

    qpu_mapping = {qpu.index: qpu for qpu in qpus}
    result_list = []
    for qpu_index, circuit_data in qpu_assignments.items():
        if circuit_data:
            results = run_circuit_list(circuit_data, backend=qpu_mapping[qpu_index].backend)
            result_list.append(results)

    variants_results = merge_and_normalize_variant_counts(result_list)
    coefficient = 1 / (16 ** num_cuts)
    # pprint(variants_results)
    res = post_processing_2.full_reconstruct(variants_results, coefficient, qc.num_qubits)
    return res


def clustered_chain_graph(n: int, r: int, k: int, q1: float, q2: float, seed: Optional[int] = None) -> Tuple[nx.Graph, List[List[int]], List[List[int]]]:
    if r <= 0 or not isinstance(r, int):
        raise ValueError("Number of clusters must be an integer greater than 0")

    clusters = []
    for i in range(r):
        _seed = seed * i if seed is not None else None
        cluster = nx.erdos_renyi_graph(n, q1, seed=_seed)
        nx.set_node_attributes(cluster, f"cluster_{i}", "subgraph")
        clusters.append(cluster)

    separators = []
    for i in range(r - 1):
        separator = nx.empty_graph(k)
        nx.set_node_attributes(separator, f"separator_{i}", "subgraph")
        separators.append(separator)

    G = nx.disjoint_union_all(clusters + separators)
    cluster_nodes = [[n[0] for n in G.nodes(data="subgraph") if n[1] == f"cluster_{i}"] for i in range(r)]
    separator_nodes = [[n[0] for n in G.nodes(data="subgraph") if n[1] == f"separator_{i}"] for i in range(r - 1)]

    rng = np.random.default_rng(seed)
    for i, separator in enumerate(separator_nodes):
        for s in separator:
            for c in cluster_nodes[i] + cluster_nodes[i + 1]:
                if rng.random() < q2:
                    G.add_edge(s, c)

    return G, cluster_nodes, separator_nodes


def get_qaoa_circuit(G: nx.Graph, cluster_nodes: List[List[int]], separator_nodes: List[List[int]], params: Tuple[Tuple[float]], layers: int = 1) -> qml.tape.QuantumTape:
    wires = len(G)
    r = len(cluster_nodes)

    with qml.tape.QuantumTape() as tape:
        for w in range(wires):
            qml.Hadamard(wires=w)

        for l in range(layers):
            gamma, beta = params[l]

            for i, c in enumerate(cluster_nodes):
                current_separator = separator_nodes[i - 1] if i > 0 else []
                next_separator = separator_nodes[i] if i < r - 1 else []
                nodes = c + current_separator + next_separator
                subgraph = G.subgraph(nodes)

                for edge in subgraph.edges:
                    qml.IsingZZ(2 * gamma, wires=edge)

            for w in range(wires):
                qml.RX(2 * beta, wires=w)

        observable = "Z" * wires
        [qml.expval(qml.pauli.string_to_pauli_word(observable))]

    return tape


def generate_qaoa_maxcut_circuit(n: int, r: int, k: int, layers: int = 1, q1: float = 0.7, q2: float = 0.3, seed: Optional[int] = None) -> str:
    G, cluster_nodes, separator_nodes = clustered_chain_graph(n, r, k, q1, q2, seed)
    params = ((0.1, 0.2), (0.3, 0.4))
    circuit = get_qaoa_circuit(G, cluster_nodes, separator_nodes, params, layers)
    return str(circuit.to_openqasm())


def run_circuit(qc, shots=1024):


    simulator = AerSimulator()
    transpiled_qc = transpile(qc, simulator)
    job = simulator.run(transpiled_qc, shots=shots)
    result = job.result()
    counts = result.get_counts()

    total_counts = sum(counts.values())
    normalized_counts = {k: v / total_counts for k, v in counts.items()}
    count_list = [0] * (2 ** len(qc.qubits))

    for k, v in normalized_counts.items():
        count_list[int(k, 2)] = v

    return count_list


def generate_circuits(filename):
    with open(filename, 'r') as file:
        data = json.load(file)

    circuits = []
    for item in data:
        circuit = generate_qaoa_maxcut_circuit(
            n=item['n'],
            r=item['r'],
            k=item['k'],
            layers=item['layers'],
            seed=item['seed']
        )
        circuits.append(circuit)

    return circuits


def hellinger_distance(p, q):
    p = np.asarray(p) / np.sum(p)
    q = np.asarray(q) / np.sum(q)
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))


if __name__ == "__main__":
    filename = './src/conf.json'
    circuits = generate_circuits(filename)
    gt = []
    res = []
    dist = []

    for i, circuit in enumerate(circuits):
        qc = QuantumCircuit.from_qasm_str(circuit)
        qc.draw('mpl')
        # plt.show()
        ground_truth = run_circuit(qc, shots=1024)
        reconstructed = test(qc)
        distance = hellinger_distance(ground_truth, reconstructed)

        gt.append((qc, ground_truth))
        res.append(reconstructed)
        dist.append(distance)


    with open("./src/results.txt", "w") as f:
        f.write("Generated circuits and their results:\n")
        for i, (qc, ground_truth) in enumerate(gt):
            f.write(f"Circuit {i + 1} results:\n")
            f.write(f"Ground truth: {ground_truth} {sum(ground_truth)}\n")
            f.write(f"Reconstructed: {res[i]} {sum(res[i])}\n")
            f.write(f"Hellinger distance: {dist[i]}\n")

import json
from typing import List, Optional, Tuple

import networkx as nx
import pennylane as qml
from pennylane import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator




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


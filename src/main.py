from matplotlib import pyplot as plt
from pprint import pprint

from qiskit import QuantumCircuit

from runner import run_circuit_list
from graph import get_dag_mapping, extract_graph_data
from merge import merge_and_normalize_variant_counts
from model import CutAndShootModel
from qiskit.converters import circuit_to_dag
from constructor import create_quantum_subcircuits, build_subcircuit_map
from qpu import QPU

def create_qpus():
    """Creates a list of QPUs and optionally overrides their metrics."""
    qpu_types = [
        'aer_simulator',
        'aer_simulator_statevector',
        'ibmq_qasm_simulator',
        'ibm_nairobi',
        'ibm_oslo',
    ]

    qpu_override_params = {
        'aer_simulator': (10, 1, 100),
        'aer_simulator_statevector': (12, 2, 100),
        'ibmq_qasm_simulator': (60, 3, 100),
        'ibm_nairobi': (3, 6, 100),
        'ibm_oslo': (5, 5, 100),
    }

    qpus = []
    for i, qpu_type in enumerate(qpu_types):
        qpu = QPU(qpu_type, i)
        if qpu_type in qpu_override_params:
            exec_time, queue_time, capacity = qpu_override_params[qpu_type]
            qpu.update_metrics(exec_time, queue_time, capacity)
        qpus.append(qpu)

    return qpus

def main():

    from generator import generate_circuits

    circuit = generate_circuits("./src/single.json")[0]
    qc = QuantumCircuit.from_qasm_str(circuit)
    # qc.draw('mpl')
    # plt.show()

    dag = circuit_to_dag(qc)
    id_mapping = get_dag_mapping(dag)
    vertex_weights, edges = extract_graph_data(dag, id_mapping)

    print(vertex_weights)
    pprint(id_mapping)

    # create QPUs
    qpus = create_qpus()

    """ print("\nNode information (using local indexes):")
    for node in dag.op_nodes():
        local_index = id_mapping[node._node_id]
        print(f"Local index: {local_index}")
        print(f"  Node: {node.name}")
        print(f"  Operation: {node.op}")
        if hasattr(node.op, "params"):
            print(f"  Parameters: {node.op.params}")
        print(f"  Qubits: {node.qargs}")
        print(f"  Condition: {node.condition}")
        print("-" * 40) """

    # solve the model using CutAndShootModel
    model = CutAndShootModel(
        edges=edges,
        vertex_weights=vertex_weights,
        qpus=qpus,
        num_shots_per_subcircuit=10000,
        num_subcircuits=4,
        alpha=0.8,
        beta=0.2
    )
    model.solve_model()
    subcircuits, num_cuts = model.print_and_return_solution()

    qc.draw('mpl')
    plt.show()

    # generate all variants of the subcircuits
    quantum_subcircuits = create_quantum_subcircuits(subcircuits, qc, id_mapping=id_mapping)

    # create mapping from node ID to DAGOpNode
    dag_nodes_dict = {
        local_index: node for node in dag.op_nodes()
        if (local_index := id_mapping.get(node._node_id)) is not None
    }

    # connection map between subcircuits for final reconstruction
    subcircuit_map = build_subcircuit_map(subcircuits, dag_nodes_dict)
    pprint(subcircuit_map)

    # asign subcircuit variants to QPUs
    qpu_assignments = {qpu.index: [] for qpu in qpus}
    for subcircuit_id, variants in quantum_subcircuits.items():
        for qpu_index, shots in subcircuits[subcircuit_id]['shots'].items():
            if shots > 0:
                for name, (circuit, active_qubits, init_qbits, measure_qbits) in variants.items():
                    qpu_assignments[qpu_index].append((name, circuit, shots, active_qubits, init_qbits, measure_qbits))

    # run each QPU's circuits and collect results
    qpu_mapping = {qpu.index: qpu for qpu in qpus}
    result_list = []
    for qpu_index, circuit_data in qpu_assignments.items():
        if not circuit_data:
            continue
        results = run_circuit_list(circuit_data, backend=qpu_mapping[qpu_index].backend)
        result_list.append(results)

    variants_results = merge_and_normalize_variant_counts(result_list)
    pprint(variants_results)

    print("Num cuts:", num_cuts)
    coefficent = 1 / pow(16, num_cuts)  # 4 init states * 4 measurement states = 16
    print("Coefficient:", coefficent)

    # final distrribution is missing

if __name__ == "__main__":
    main()

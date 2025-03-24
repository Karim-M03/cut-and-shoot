import examples.grover
import examples.rca
import examples.simple_circuit
from runner import run_circuit_list
from graph import get_dag_mapping, extract_graph_data
from merge import merge_and_normalize_variant_counts


from model import CutAndShootModel
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from constructor import create_quantum_subcircuits
from qpu import QPU
from pprint import pprint



def create_qpus():
    """Creates a list of QPUs and optionally overrides their metrics."""

    qpu_types = [
        'aer_simulator',
        'aer_simulator_statevector',
        'ibmq_qasm_simulator',
        'ibm_nairobi',
        'ibm_oslo',
    ]

    # optional override: (qpu_type, execution_time, queue_time, capacity)
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
        if qpu_type in qpu_override_params:
            exec_time, queue_time, capacity = qpu_override_params[qpu_type]
            qpu.update_metrics(exec_time, queue_time, capacity)
        qpus.append(qpu)

    return qpus




def main():
    # choose a circuit example
    # qc = examples.simple_circuit.simple_circuit()
    # qc = examples.rca.ripple_carry_adder(2)
    qc = examples.grover.grover_circuit(2)
    print(qc.draw())

    # convert to DAG and extract data
    dag = circuit_to_dag(qc)
    id_mapping = get_dag_mapping(dag)
    vertex_weights, edges = extract_graph_data(dag, id_mapping)

    print(vertex_weights)
    pprint(id_mapping)

    # create QPUs and model
    qpus = create_qpus()
    qpu_params = [
        ('aer_simulator', 10, 1, 10),
        ('aer_simulator_statevector', 12, 2, 8),
        ('ibmq_qasm_simulator', 60, 3, 6),
        ('ibm_nairobi', 300, 6, 2),
        ('ibm_oslo', 280, 5, 3),
    ]

    pprint(qpus)

    # print local index and node information for each DAG operation node
    print("\nNode information (using local indexes):")
    for node in dag.op_nodes():
        local_index = id_mapping[node._node_id]
        print(f"Local index: {local_index}")
        print(f"  Node: {node.name}")
        print(f"  Operation: {node.op}")
        if hasattr(node.op, "params"):
            print(f"  Parameters: {node.op.params}")
        print(f"  Qubits: {node.qargs}")
        print(f"  Condition: {node.condition}")
        print("-" * 40)

    vertex_weights, edges = extract_graph_data(dag, id_mapping)

    #change the data according to the requirements
    model = CutAndShootModel(
        edges=edges,
        vertex_weights=vertex_weights,
        qpus=qpus,
        num_shots_per_subcircuit=100,
        num_subcircuits=4,
        alpha=0.8,
        beta=0.2
    )

    model.solve_model()
    subcircuits = model.print_and_return_solution()

    # create all subcircuit variants
    quantum_subcircuits = create_quantum_subcircuits(subcircuits, qc, id_mapping=id_mapping)

    # group subcircuits per QPU
    qpu_assignments = {qpu.index: [] for qpu in qpus}

    for subcircuit_id, variants in quantum_subcircuits.items():
        for qpu_index, shots in subcircuits[subcircuit_id]['shots'].items():
            if shots > 0:
                for name, (circuit, active_qubits) in variants.items():
                    qpu_assignments[qpu_index].append((name, circuit, shots, active_qubits))


    pprint(qpu_assignments)
    
    # create mapping from qpu index to qpu object
    qpu_mapping = {qpu.index: qpu for qpu in qpus}
    result_list = []

    # run each QPU's assigned circuits
    for qpu_index, circuit_data in qpu_assignments.items():
        if not circuit_data or len(circuit_data) == 0:
            continue

        qpu = qpu_mapping.get(qpu_index)
        print(f"\n========== Executing circuits for QPU {qpu_index} ({qpu}) ==========\n")

        results = run_circuit_list(circuit_data, backend=qpu_mapping[qpu_index].backend)
        pprint(results)
        result_list.append(results)

    pprint(merge_and_normalize_variant_counts(result_list))


    

if __name__ == "__main__":
    main()

import logging
import pprint
from matplotlib import pyplot as plt

from qiskit import QuantumCircuit

from formatter import format_data
from runner import run_circuit_list
from graph import get_dag_mapping, extract_graph_data
from merge import merge_and_normalize_variant_counts
from model import CutAndShootModel
from qiskit.converters import circuit_to_dag
from constructor import create_quantum_subcircuits
from qpu import QPU
from logger import get_logger


DEBUG = True
logger = get_logger("main_logger", "./src/output/main.log", level=logging.DEBUG)
pp = pprint.PrettyPrinter(indent=2)

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
        'aer_simulator': (10, 1, 70),
        'aer_simulator_statevector': (12, 2, 70),
        'ibmq_qasm_simulator': (70, 3, 70),
        'ibm_nairobi': (3, 6, 50),
        'ibm_oslo': (100, 100, 100),
    }

    qpus = []
    for i, qpu_type in enumerate(qpu_types):
        qpu = QPU(qpu_type, i)
        if qpu_type in qpu_override_params:
            exec_time, queue_time, capacity = qpu_override_params[qpu_type]
            qpu.update_metrics(exec_time, queue_time, capacity)
        qpus.append(qpu)
    
    logger.info(f"Created {len(qpus)} QPUs")
    return qpus

def main():

    from generator import generate_circuits

    circuit = generate_circuits("./src/test.json")[0]
    qc = QuantumCircuit.from_qasm_str(circuit)
    logger.debug("Loaded and parsed quantum circuit")
    # qc.draw('mpl')
    # plt.show()
    logger.debug(qc.draw(output="text").single_string())

    # qc = simple_circuit.simple_circuit()
    dag = circuit_to_dag(qc)
    id_mapping = get_dag_mapping(dag)
    vertex_weights, edges = extract_graph_data(dag, id_mapping)
    logger.debug("Extracted graph data from DAG")

    logger.debug("Weights:\n%s", pp.pformat(vertex_weights))
    logger.debug("ID Mapping:\n%s", pp.pformat(id_mapping))

    # create QPUs
    qpus = create_qpus()

    logger.debug("\nNode information (using local indexes):")
    for node in dag.op_nodes():
        local_index = id_mapping[node._node_id]
        logger.debug(f"Local index: {local_index}")
        logger.debug(f"  Node: {node.name}")
        logger.debug(f"  Operation: {node.op}")
        if hasattr(node.op, "params"):
            logger.debug(f"  Parameters: {node.op.params}")
        logger.debug(f"  Qubits: {node.qargs}")
        logger.debug(f"  Condition: {node.condition}")
        logger.debug("-" * 40)


    # solve the model using CutAndShootModel
    model = CutAndShootModel(
        edges=edges,
        vertex_weights=vertex_weights,
        qpus=qpus,
        num_shots_per_subcircuit=10000,
        num_subcircuits=4,
        alpha=0.5,
        beta=0.5
    )
    model.solve_model()
    subcircuits, num_cuts = model.print_and_return_solution()
    
    # generate all variants of the subcircuits
    quantum_subcircuits = create_quantum_subcircuits(subcircuits, qc, id_mapping=id_mapping)
    
    for sub_id, sub_variants_dict in quantum_subcircuits.items():
        logger.debug(f"================ Subcircuit {sub_id} ================\n")
        for variant_name, variant in sub_variants_dict.items():
            logger.debug(f"\n{variant_name}\n")
            logger.debug(f"\n{variant.__repr__()}\n")
            logger.debug(variant.circuit.draw(output="text").single_string())
            logger.debug("\n\n")

    # asign subcircuit variants to QPUs
    qpu_assignments = {qpu.index: [] for qpu in qpus}
    for subcircuit_id, variants in quantum_subcircuits.items():
        for qpu_index, shots in subcircuits[subcircuit_id]['shots'].items():
            if shots > 0:
                for name, variant in variants.items():
                    variant.shots = shots
                    qpu_assignments[qpu_index].append(variant)

    logger.debug("Assigned subcircuits to QPUs")

    # run each QPU's circuits and collect results
    qpu_mapping = {qpu.index: qpu for qpu in qpus}
    result_list = []
    for qpu_index, circuit_data in qpu_assignments.items():
        if not circuit_data:
            continue
        logger.info(f"Running {len(circuit_data)} variants on QPU {qpu_index}")
        results = run_circuit_list(circuit_data, backend=qpu_mapping[qpu_index].backend)
        result_list.append(results)

    logger.debug("Collected results from all QPUs")

    variants_results = merge_and_normalize_variant_counts(result_list)
    logger.info("Merged and normalized variant counts")

    formatted_data = format_data(quantum_subcircuits, variants_results)
    logger.debug("Formatted final data for reconstruction")

    logger.info("Execution completed successfully")
    logger.debug(pp.pformat(formatted_data))



if __name__ == "__main__":
    main()

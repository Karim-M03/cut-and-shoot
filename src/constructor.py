import itertools
import logging
from collections import defaultdict, deque
from pprint import pprint
from queue import Queue
from typing import Dict, List, Tuple, Deque, Any

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit.dagnode import DAGOpNode
from concurrent.futures import ThreadPoolExecutor, as_completed
from subcircuit import Variant
from logger import get_logger
import time


# possible initial states
init_states: Dict[str, List[complex]] = {
    '|0>': [1, 0],
    '|1>': [0, 1],
    '|+>': [1 / np.sqrt(2), 1 / np.sqrt(2)],
    '|i>': [1 / np.sqrt(2), 1j / np.sqrt(2)]
}

def calculate_required_qubits(
    vertices: set,
    dag_nodes: List[DAGOpNode], # type: ignore
    id_mapping: Dict[int, int],
) -> Tuple[int, Dict[Any, int]]:
    """
    determines the set of physical qubits used in a subset of DAG nodes and
    assigns each a local index for subcircuit creation.

    :param vertices: set of vertex IDs belonging to the subcircuit
    :param dag_nodes: list of all DAGOpNodes from the full circuit
    :param id_mapping: mapping from original DAG node IDs to normalized node IDs
    :return (number of required qubits, mapping from global to local qubit index)
    """

    # to avoid duplicates
    qubits_used = set()
    
    # check which physical qubits are actually used by the selected nodes
    for node in dag_nodes:
        mapped_id = id_mapping[node._node_id]
        if mapped_id in vertices:
            qubits_used.update(node.qargs)

    # sort them by register name and index for consistency
    sorted_qubits = sorted(qubits_used, key=lambda q: (q._register.name, q._index))
    qubit_mapping = {qb: local_idx for local_idx, qb in enumerate(sorted_qubits)}
    return len(sorted_qubits), qubit_mapping

def apply_initializations(
    sub_qc: QuantumCircuit,
    qreg: QuantumRegister,
    qubit_indices: List[int],
    init_variant: str,
)  -> Dict[int, str]:
    """
    applies a specific initial state to selected qubits in the subcircuit
    and returns the list of (qubit_index, basis) for naming purposes.

    :return: list of initialized qubits with their basis
    """
    state = init_states[init_variant]
    initialized = {}
    for idx in qubit_indices:
        sub_qc.initialize(state, qreg[idx])
        initialized[idx] = init_variant

    return initialized


def append_gates(
    sub_qc: QuantumCircuit,
    dag_nodes: List[DAGOpNode], # type: ignore
    vertices: set,
    qreg: QuantumRegister,
    init_map: Dict[Any, str],
    id_mapping: Dict[int, int],
    qbit_mapping: Dict[Any, int],
)  -> Dict[int, str]:
    """
    adds all relevant gates to the subcircuit and handles qubit initialization
    at cut points

    :param sub_qc: the subcircuit being constructed
    :param dag_nodes: full list of DAGOpNodes
    :param vertices: vertex ids belonging to the current subcircuit
    :param qreg: quantum register for the subcircuit
    :param init_map: maps qbit -> initialization
    :param id_mapping: mapping from original dag node ids to local node ids
    :param qbit_mapping: mapping from global qubits to local indices
    """

    initialized = {}
    already_initted = set()  # track which qubits have been initialized

    for node in dag_nodes:
        mapped_id_node = id_mapping[node._node_id]
        # skip if it's not part of this subcircuit or it's a measurement node
        if mapped_id_node not in vertices or node.op.name.lower() == 'measure':
            continue

        # if a qubit needs initialization and hasn t been initialized yet do so once.
        for qb in node.qargs:
            if qb in init_map and qb not in already_initted:
                idx = qbit_mapping[qb]
                init_dict = apply_initializations(sub_qc, qreg, [idx], init_map[qb])
                initialized.update(init_dict)
                already_initted.add(qb)


        # after possibly initializing any required qubits, append the gate only once
        sub_qc.append(node.op, [qreg[qbit_mapping[q]] for q in node.qargs])

    return initialized

def append_measurements(
    sub_qc: QuantumCircuit,
    qreg: QuantumRegister,
    unique_qbs: List[Any],
    out_combo: Tuple[str, ...],
    qbit_mapping: Dict[Any, int],
) -> Dict[int, str]:
    """
    adds measurement operations to the subcircuit according to the provided
    measurement basis for each cut out point.

    :param sub_qc: the subcircuit being constructed
    :param qreg: quantum register for the subcircuit
    :param dag_nodes_dict: mapping from node ID to DAGOpNode
    :param cuts_out: list of output cut node IDs
    :param out_combo: tuple indicating the basis for each cut
    :param qbit_mapping: mapping from global qubits to local indices
    """
    measured = {}
    cbit_index = 0

    for qb, basis in zip(unique_qbs, out_combo):
        local_idx = qbit_mapping[qb]

        if basis == 'X':
            sub_qc.h(qreg[local_idx])
        elif basis == 'Y':
            sub_qc.sdg(qreg[local_idx])
            sub_qc.h(qreg[local_idx])
        # Z/I: no transformation

        measured[local_idx] = basis
        cbit_index += 1

    return measured

def create_quantum_subcircuits(
    subcircuits: Dict[int, Dict[str, object]],
    qc: QuantumCircuit,
    id_mapping: Dict[int, int],
    max_workers: int = 8
) ->  Dict[int, Dict[str, Variant]]:
    """
    generates all variants of the quantum subcircuits defined in subcircuits,
    accounting for all combinations of input initializations and output measurements

    :param subcircuits: dictionary mapping subcircuit id to subgraph structure:
                          {
                            id: {
                              "vertices": [...],
                              "cuts": {
                                "in": [...],
                                "out": [...]
                              }
                              "cuts_info:{
                                "in": [...],
                                "out": [...],
                              }
                            }
                          }
    :param qc: the original  quantum circuit needed to know which node do waht
    :param id_mapping: mapping from original dag node ids to local node ids
    :param max_workers: max number of threads
    :return dictionary {sub_id: {variant_name: Variant, ...}}
    """
    
    result: Dict[int, Dict[str, Variant]] = defaultdict(dict)

    # convert full circuit to dag for node access
    dag = circuit_to_dag(qc)
    dag_nodes = list(dag.op_nodes())
    # build a dict for quick node lookup
    dag_nodes_dict = {id_mapping[node._node_id]: node for node in dag_nodes}

    # function to generate all circuit variants for a single subcircuit
    def generate_subcircuits(sub_id: int, subcircuit: Dict[str, object]):

        logger = get_logger(f"Sub {sub_id} Logger", f"./src/output/constructor/{time.time()}sub_{sub_id}.log" )
        local_subcircuits = {}

        # extract subcircuit info
        vertices = set(subcircuit['vertices'])
        cuts_in = sorted([n for n in subcircuit['cuts']['in'] if dag_nodes_dict[n].op.name.lower() != 'barrier'])
        cuts_out = sorted([n for n in subcircuit['cuts']['out'] if dag_nodes_dict[n].op.name.lower() != 'barrier'])

        unique_cut_in_qubits = sorted({
            qb for node_id in cuts_in
            for qb in dag_nodes_dict[node_id].qargs
        }, key=lambda q: (q._register.name, q._index))

        all_in_combos = list(itertools.product(['|0>', '|1>', '|+>', '|i>'], repeat=len(unique_cut_in_qubits)))

        # get all the qbits involved in cut_out nodes
        unique_qbs = sorted({
            qb for node_id in cuts_out
            for qb in dag_nodes_dict[node_id].qargs
        }, key=lambda q: (q._register.name, q._index))

        all_out_combos = list(itertools.product(['I', 'X', 'Z', 'Y'], repeat=len(unique_qbs)))
        logger.info(f"Generating subcircuit {sub_id} with {len(all_in_combos)} in-combos and {len(all_out_combos)} out-combos")


        # loop through each input combo
        for in_combo in all_in_combos:
            num_qubits, qbit_map = calculate_required_qubits(vertices, dag_nodes, id_mapping)
            qreg = QuantumRegister(num_qubits, f"q{sub_id}")
            base_sub_qc = QuantumCircuit(qreg)

            # initialize cut_in states
            init_map = {
                qb: basis for qb, basis in zip(unique_cut_in_qubits, in_combo)
            }

            initialized_info = append_gates(
                base_sub_qc,
                dag_nodes,
                vertices,
                qreg,
                init_map,
                id_mapping,
                qbit_map,
            )

            # for naming
            initialized_qbs_str = "_".join([f"q{q}-{b}" for q, b in initialized_info.items()])
            
            for out_combo in all_out_combos:
                sub_qc_variant = base_sub_qc.copy()
                
                unique_qbs = sorted({
                    qb for node_id in cuts_out
                    for qb in dag_nodes_dict[node_id].qargs
                }, key=lambda q: (q._register.name, q._index))
                

                measured_info = append_measurements(
                    sub_qc_variant,
                    qreg,
                    unique_qbs,
                    out_combo,
                    qbit_map,
                )

                # measured_qubits = sorted({qbit_map[qb] for qb in unique_qbs})
                active_qubits = sorted(qbit_map.values())

                local_to_global_qbit_map = {
                    local_idx: dag.qubits.index(qb)
                    for qb, local_idx in qbit_map.items()
                }

                measured_qbs_str = "_".join([f"q{q}-{b}" for q, b in measured_info.items()])


                # name and store the variant
                circuit_name = f"sub_{sub_id}_in_{initialized_qbs_str}_out_{measured_qbs_str}"

                local_subcircuits[circuit_name] = Variant(
                    sub_id=sub_id, 
                    shots=subcircuit["shots"],
                    name=circuit_name,
                    vertices=vertices,
                    cuts_info=subcircuit["cuts_info"],
                    circuit=sub_qc_variant,
                    active_qubits=active_qubits,
                    initialized_info=initialized_info,
                    measured_info=measured_info,
                    qbit_map=local_to_global_qbit_map
                )
                logger.info(f"Created variant {circuit_name}")

        logger.info(f"Subcircuit {sub_id}: generated {len(local_subcircuits)} variants.")
        return sub_id, local_subcircuits
    
    # generate all subcircuits in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_subcircuits, sub_id, subcircuit)
                   for sub_id, subcircuit in subcircuits.items()]
        for future in as_completed(futures):
            sub_id, local_subcircuits = future.result()
            result[sub_id] = local_subcircuits


    return result

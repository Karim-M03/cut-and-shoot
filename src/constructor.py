import itertools
import logging
from collections import defaultdict, deque
from pprint import pprint
from typing import Dict, List, Tuple, Deque, Any

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit.dagnode import DAGOpNode
from concurrent.futures import ThreadPoolExecutor, as_completed




# possible initial states
init_states: Dict[str, List[complex]] = {
    '0': [1, 0],
    '1': [0, 1],
    '+': [1 / np.sqrt(2), 1 / np.sqrt(2)],
    'i': [1 / np.sqrt(2), 1j / np.sqrt(2)]
}

def calculate_required_qubits(
    vertices: set,
    dag_nodes: List[DAGOpNode],
    id_mapping: Dict[int, int]
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
    init_variant: str
) -> None:
    """
    applies a specific initial state to selected qubits in the subcircuit.

    :param sub_qc: the subcircuit being constructed
    :param qreg: quantum register used in the subcircuit
    :param qubit_indices: list of qubit indices to initialize
    :param init_variant: initialization type
    """
    # get state vector for the selected initialization
    state = init_states[init_variant]

    # apply it to each local qubit index
    for idx in qubit_indices:
        sub_qc.initialize(state, qreg[idx])

def append_gates(
    sub_qc: QuantumCircuit,
    dag_nodes: List[DAGOpNode],
    vertices: set,
    qreg: QuantumRegister,
    cuts_in: set,
    init_variants_queue: Dict[int, Deque[str]],
    id_mapping: Dict[int, int],
    qbit_mapping: Dict[Any, int]
) -> None:
    """
    adds all relevant gates to the subcircuit and handles qubit initialization
    at cut points

    :param sub_qc: the subcircuit being constructed
    :param dag_nodes: full list of DAGOpNodes
    :param vertices: vertex ids belonging to the current subcircuit
    :param qreg: quantum register for the subcircuit
    :param cuts_in: ids of input-cut vertices
    :param init_variants_queue: queue of initialization variants for cut_ins
    :param id_mapping: mapping from original dag node ids to local node ids
    :param qbit_mapping: mapping from global qubits to local indices
    """
 
    for node in dag_nodes:
        mapped_id_node = id_mapping[node._node_id]
        # skip if it's not part of this subcircuit or it's a measurement node
        if mapped_id_node not in vertices or node.op.name.lower() == 'measure':
            continue

        # map global qubits to local indices
        local_indices = [qbit_mapping[qb] for qb in node.qargs]

        # handle input cut initialization
        if mapped_id_node in cuts_in and init_variants_queue[mapped_id_node]:
            variant = init_variants_queue[mapped_id_node].popleft()
            apply_initializations(sub_qc, qreg, local_indices, variant)
        else:
            # no duplicate qubitsi
            if len(set(local_indices)) < len(local_indices):
                print('Error: node has duplicate qubits:', node.op.name, node.qargs)
            sub_qc.append(node.op, [qreg[idx] for idx in local_indices])

def append_measurements(
    sub_qc: QuantumCircuit,
    qreg: QuantumRegister,
    creg: ClassicalRegister,
    dag_nodes_dict: Dict[int, DAGOpNode],
    cuts_out: List[int],
    out_combo: Tuple[str, ...],
    qbit_mapping: Dict[Any, int]
) -> None:
    """
    adds measurement operations to the subcircuit according to the provided
    measurement basis for each cut out point.

    :param sub_qc: the subcircuit being constructed
    :param qreg: quantum register for the subcircuit
    :param creg: classical register for measurements
    :param dag_nodes_dict: mapping from node ID to DAGOpNode
    :param cuts_out: list of output cut node IDs
    :param out_combo: tuple indicating the basis for each cut
    :param qbit_mapping: mapping from global qubits to local indices
    """
    cbit_index = 0
    measured_qubits = set()

     # loop through each output node and apply the measurement
    for out_node_id, measure_variant in zip(cuts_out, out_combo):
        out_dag_node = dag_nodes_dict.get(out_node_id)
        if not out_dag_node:
            continue

        # sort qubits for consistency
        sorted_qbs = sorted(out_dag_node.qargs, key=lambda q: (q._register.name, q._index))
        for qb in sorted_qbs:
            local_qb_idx = qbit_mapping[qb]

            #skip if already measured
            if local_qb_idx in measured_qubits:
                continue

            # apply the correct basiss transformation
            if measure_variant == 'id':
                # no rotation
                pass
            elif measure_variant == 'x':
                sub_qc.h(qreg[local_qb_idx])
            elif measure_variant == 'y':
                sub_qc.sdg(qreg[local_qb_idx])
                sub_qc.h(qreg[local_qb_idx])
            elif measure_variant == 'z':
                pass
            
            # perform the measurement
            sub_qc.measure(qreg[local_qb_idx], creg[cbit_index])
            cbit_index += 1
            measured_qubits.add(local_qb_idx)

def create_quantum_subcircuits(
    subcircuits: Dict[int, Dict[str, object]],
    qc: QuantumCircuit,
    id_mapping: Dict[int, int],
    max_workers: int = 8
) -> Dict[int, Dict[str, Tuple[QuantumCircuit, List[int], List[int], List[int]]]]:
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
                            }
                          }
    :param qc: the original  quantum circuit needed to know which node do waht
    :param id_mapping: mapping from original dag node ids to local node ids
    :param max_workers: max number of threads
    :return dictionary {sub_id: {variant_name: QuantumCircuit, ...}, ...}
    """
    
    result: Dict[int, Dict[str, Tuple[QuantumCircuit, List[int], List[int], List[int]]]] = defaultdict(dict)

    # convert full circuit to dag for node access
    dag = circuit_to_dag(qc)
    dag_nodes = list(dag.op_nodes())
    # build a dict for quick node lookup
    dag_nodes_dict = {id_mapping[node._node_id]: node for node in dag_nodes}

    # function to generate all circuit variants for a single subcircuit
    def generate_subcircuits(sub_id: int, subcircuit: Dict[str, object]):
        local_subcircuits = {}

        # extract subcircuit info
        vertices = set(subcircuit['vertices'])
        cuts_in = [n for n in subcircuit['cuts']['in'] if dag_nodes_dict[n].op.name.lower() != 'barrier']
        cuts_out = [n for n in subcircuit['cuts']['out'] if dag_nodes_dict[n].op.name.lower() != 'barrier']

        # prepare all combinations of inputs and measurement bases
        all_in_combos = list(itertools.product(['0', '1', '+', 'i'], repeat=len(cuts_in)))
        all_out_combos = list(itertools.product(['id', 'x', 'z', 'y'], repeat=len(cuts_out)))

        # loop through each input combo
        for in_combo in all_in_combos:
            num_qubits, qbit_map = calculate_required_qubits(vertices, dag_nodes, id_mapping)
            qreg = QuantumRegister(num_qubits, f"q{sub_id}")
            base_sub_qc = QuantumCircuit(qreg)

            # initialize cut_in states
            init_variants_queue = {
                node_id: deque([variant])
                for node_id, variant in zip(cuts_in, in_combo)
            }

            append_gates(
                base_sub_qc,
                dag_nodes,
                vertices,
                qreg,
                set(cuts_in),
                init_variants_queue,
                id_mapping,
                qbit_map
            )

            initialized_qubits = sorted({qbit_map[qb] for cut in cuts_in for qb in dag_nodes_dict[cut].qargs})

            
            for out_combo in all_out_combos:
                sub_qc_variant = base_sub_qc.copy()

                unique_qbs = {
                    qb
                    for out_node_id in cuts_out
                    for qb in dag_nodes_dict[out_node_id].qargs
                }
                creg = ClassicalRegister(len(unique_qbs), f"meas_{sub_id}")
                sub_qc_variant.add_register(creg)

                append_measurements(
                    sub_qc_variant,
                    qreg,
                    creg,
                    dag_nodes_dict,
                    cuts_out,
                    out_combo,
                    qbit_map
                )

                measured_qubits = sorted({qbit_map[qb] for qb in unique_qbs})
                active_qubits = sorted(qbit_map.values())

                # name and store the variant
                circuit_name = f"sub_{sub_id}_in_{'-'.join(in_combo)}_out_{'-'.join(out_combo)}"
                local_subcircuits[circuit_name] = (
                    sub_qc_variant,
                    active_qubits,
                    initialized_qubits,
                    measured_qubits
                )

        print(f'Subcircuit {sub_id}: generated {len(local_subcircuits)} variants.')
        return sub_id, local_subcircuits
    
    # generate all subcircuits in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_subcircuits, sub_id, subcircuit)
                   for sub_id, subcircuit in subcircuits.items()]
        for future in as_completed(futures):
            sub_id, local_subcircuits = future.result()
            result[sub_id] = local_subcircuits

    pprint(result)
    return result



def build_subcircuit_map(
    subcircuits: Dict[int, Dict[str, Any]],
    dag_nodes_dict: Dict[int, DAGOpNode]
) -> Dict[Tuple[int, int], List[List[str]]]:
    """
    builds a compact mapping of subcircuits that are connected through shared qubits
    Returns a linear dictionary of the form:
        {(sub_from, sub_to): [[shared_qb_out, shared_qb_in], ...]}
    
    Each key represents a directional connection from one subcircuit to another,
    and the value is a list of shared qubit names involved in the cut.
    """

    # extract output cut qubits for each subcircuit, ignoring barrier nodes
    out_cuts_info = {}
    for sub_id, data in subcircuits.items():
        out_nodes = [n for n in data['cuts']['out'] if dag_nodes_dict[n].op.name.lower() != 'barrier']
        out_qubits = []
        for node_id in out_nodes:
            qubits = sorted(dag_nodes_dict[node_id].qargs, key=lambda q: (q._register.name, q._index))
            out_qubits.append(qubits)
        out_cuts_info[sub_id] = out_qubits

    # extract input cut qubits for each subcircuit, ignoring barrier nodes
    in_cuts_info = {}
    for sub_id, data in subcircuits.items():
        in_nodes = [n for n in data['cuts']['in'] if dag_nodes_dict[n].op.name.lower() != 'barrier']
        in_qubits = []
        for node_id in in_nodes:
            qubits = sorted(dag_nodes_dict[node_id].qargs, key=lambda q: (q._register.name, q._index))
            in_qubits.append(qubits)
        in_cuts_info[sub_id] = in_qubits

    # build the compact connection map between subcircuits
    compact_map: Dict[Tuple[int, int], List[List[str]]] = {}

    for sub_id_out, out_qubit_lists in out_cuts_info.items():
        for sub_id_in, in_qubit_lists in in_cuts_info.items():
            if sub_id_out == sub_id_in:
                continue  # skip self connections

            # compare each out qubit list with each in qubit list
            for out_qubits in out_qubit_lists:
                for in_qubits in in_qubit_lists:
                    # find shared qubits between output and input cuts
                    shared = sorted(set(out_qubits) & set(in_qubits), key=lambda q: (q._register.name, q._index))
                    if shared:
                        key = (sub_id_out, sub_id_in)
                        if key not in compact_map:
                            compact_map[key] = []

                        shared_str = [f"{q._register.name}[{q._index}]" for q in shared]
                        compact_map[key].append(shared_str)

    return compact_map

from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from typing import Dict, Tuple, List, Set


def get_dag_mapping(dag: DAGCircuit) -> Dict[int, int]:
    """
    remap DAG node IDs to local indices starting from 0
    """
    return {node._node_id: new_id for new_id, node in enumerate(dag.topological_op_nodes())}


def extract_graph_data(
    dag: DAGCircuit,
    id_mapping: Dict[int, int]
) -> Tuple[Dict[int, int], List[Tuple[int, int]]]:
    """
    extract vertex weights and edges from a DAGCircuit using a node ID mapping.

    :param dag: Qiskit DAGCircuit.
    :param id_mapping: mapping from original DAG node IDs to local IDs.

    :returns: A tuple containing:
         - vertex_weights: a dictionary mapping node ID to its weight (number of qargs)
        - edges: edgeg of the dag using local indices
    """
    vertex_weights: Dict[int, int] = {
        id_mapping[node._node_id]: len(node.qargs)
        for node in dag.topological_op_nodes()
    }

    edges: Set[Tuple[int, int]] = {
        (id_mapping[u._node_id], id_mapping[v._node_id])
        for u, v, _ in dag.edges()
        if isinstance(u, DAGOpNode) and isinstance(v, DAGOpNode)
    }

    return vertex_weights, list(edges)

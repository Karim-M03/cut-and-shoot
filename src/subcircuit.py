from qiskit import QuantumCircuit


class Subcircuit:
    def __init__(self, sub_id: int, vertices: set, cuts_info: dict, qbit_map: dict, shots: int):
        self.sub_id = sub_id
        self.vertices = vertices
        self.cuts_info = cuts_info    # {'in': [...], 'out': [...]}
        self.qbit_map = qbit_map
        self.shots = shots


    def __repr__(self):
        return f"Subcircuit(id={self.sub_id}, vertices={list(self.vertices)}, cuts={self.cuts})"

class Variant(Subcircuit):
    def __init__(
        self,
        sub_id: int,
        shots: int,
        name: str,
        vertices: set,
        cuts_info: dict,
        circuit: QuantumCircuit,
        active_qubits: list,
        initialized_info: list,
        measured_info: list,
        qbit_map: dict
    ):
        super().__init__(sub_id, vertices, cuts_info, qbit_map, shots)
        self.circuit = circuit
        self.name = name
        self.active_qubits = active_qubits
        self.initialized_info = initialized_info
        self.measured_info = measured_info

    def __repr__(self):
        return (
            f"Variant(\n"
            f"  sub_id={self.sub_id},\n"
            f"  name='{self.name}',\n"
            f"  active_qubits={self.active_qubits},\n"
            f"  initialized_info={self.initialized_info},\n"
            f"  measured_info={self.measured_info},\n"
            f"  num_vertices={len(self.vertices)}\n"
            f"  map={self.qbit_map}\n"
            f")"
        )

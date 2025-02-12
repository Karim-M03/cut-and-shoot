

from final_model.model import ModelloCutAndShoot
from circuit.graph_to_circuit import build_subcircuits
from qpu.qpu import QPU
import pennylane as qml
from pennylane import CircuitGraph
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx  




# Dispositivo simulato a 5 qubit
dev = qml.device("default.qubit", wires=5)

@qml.qnode(dev)
def circuito():
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
    qml.Hadamard(wires=2)
    qml.Hadamard(wires=3)
    qml.Hadamard(wires=4)
    qml.Toffoli(wires=[0, 1, 2])    
    qml.CNOT(wires=[2, 3])
    qml.T(wires=2)
    qml.RX(np.pi/2, wires=0)
    qml.RY(np.pi/2, wires=4)
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
    qml.Hadamard(wires=2)
    qml.Hadamard(wires=3)
    qml.Hadamard(wires=4)

    return [qml.expval(qml.PauliZ(i)) for i in range(5)]




circuito.construct((), {})
tape = circuito.qtape

circuit_graph = CircuitGraph(tape.operations, tape.observables, tape.wires)
dag = circuit_graph.graph
vertices = dag.nodes()
edges = dag.edge_list()



vertex_weights = {}
for node in dag.nodes():
    if node < len(tape.operations):
        vertex_weights[node] = len(tape.operations[node].wires)
    else:
        indice_osservabile = node - len(tape.operations)
        vertex_weights[node] = len(tape.observables[indice_osservabile].wires)

print(set(vertices))
print(list(edges))
print(vertex_weights)




tipi = ['default.mixed', 'default.qubit', 'default.qubit', 'default.mixed', 'default.mixed']
tempo_di_esecuzione = [10, 200, 150, 300, 10] # di un singolo shot su una qpu
tempo_di_coda       = [ 1,  4,  3,  1,  3]
capacita            = [ 5,  5,  5,  5,  5]

qpus = []
for i in range(len(tipi)):
    qpus.append(QPU(tipi[i], tempo_di_esecuzione[i], tempo_di_coda[i], capacita[i], i))

# creazione del modello
modello = ModelloCutAndShoot(
    edges=edges,
    vertex_weights=vertex_weights,
    qpus=qpus,
    num_shots_per_subcircuit=100,
    num_subcircuits=4,
    alpha=0.8,
    beta=0.2
)

# eisoluzione
status, obj_value = modello.solve_model()

# stampa risultati
sottocircuiti = modello.stampa_e_restituisci_risultato()
print("--------------------------------")
circuit_queue = build_subcircuits(circuito.qtape, sottocircuiti, qpus)
circuit_queue.stampa_coda()
print("--------------------------------")
circuit_queue.esegui_sottocircuiti()

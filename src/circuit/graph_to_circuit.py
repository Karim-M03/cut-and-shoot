import pennylane as qml
from .queue import CircuitQueue

def costruisci_sottocircuiti(original_tape, sottocircuiti, qpus):
    """
    costruisce i sottocircuiti in forma di QuantumTape 
    li aggiunge alla coda associandoli al numero di shots
    
    :param original_tape: i taape iniziale che contiene le operazioni e osservabili globali
    :param sottocircuiti: dizionario con chiave = id sottocircuito, valore = {"vertices": [...], "shots": {...}}
    :param qpus: lista di QPU
    
    :return: istanza di CircuitQueue con tutti i sottocircuiti aggiunti
    """
    circuit_queue = CircuitQueue(qpus)
    
    for sub_id, info in sottocircuiti.items():
        vertices = info["vertices"]
        shots_info = info["shots"] 

        # se non ci sono vertici considera vuoto
        if not vertices or len(vertices) == 0:
            continue

        # per ogni QPU, controlliamo se è previsto un numero di shot
        for q in qpus:
            if q.index not in shots_info:
                continue

            num_shots = shots_info[q.index]

            # dividi osservabili e operazioni
            ops_idx = [v for v in vertices if v < len(original_tape.operations)]
            obs_idx = [v - len(original_tape.operations) for v in vertices if v >= len(original_tape.operations)]
            
            sub_ops = [original_tape.operations[i] for i in ops_idx]
            sub_obs = [original_tape.observables[i] for i in obs_idx]
            
            print(f"Aggiunta sottocircuito {sub_id} alla  {q.index}")
            circuit_queue.aggiungi_sottocircuito(q.index, sub_ops, sub_obs, num_shots)

    return circuit_queue
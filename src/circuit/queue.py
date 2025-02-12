import random
import pennylane as qml

class CircuitQueue:
    def __init__(self, original_tape, sottocircuiti, qpus):
        # inizializza la struttura 
        # dizionario {qpu_index_0: [QuantumTape], qpu_index_1: [QuantumTape], ...}
        subcircuit_queue = {}
        for q in qpus:
            subcircuit_queue[q.index] = []

        self.subcircuit_queue = subcircuit_queue
        self.qpus = qpus

        #costruisci i sottocircuiti
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
                self.aggiungi_sottocircuito(q.index, sub_ops, sub_obs, num_shots)   



        

    def aggiungi_sottocircuito(self, qpu_index, ops, obs, n_shots):
        """
        Crea un QuantumTape con le operazioni e gli osservabili
        e lo memorizza in subcircuit_queue insieme al numero di shot n_shots.
        
        :param qpu_index: Indice della QPU su cui verrà eseguito il sottocircuito
        :param ops: lista di operazioni del sottocircuito
        :param obs: lista di osservabili del sottocircuito
        :param n_shots: numero di shot del sottocircuito
        """
        with qml.tape.QuantumTape() as tape:
            # aggiunge le operazioni
            for op in ops:
                qml.apply(op)
            
            # simula rumore
            for q in self.qpus:
                if q.index == qpu_index and q.tipo == "default.mixed":
                    random_wire = random.randint(0, q.capacita - 1)                     
                    qml.DepolarizingChannel(0.01, wires=random_wire)
                    break
            # osservazioni
            for ob in obs:
                qml.expval(ob)
        
        # memorizza il circuito con il numero di shot
        self.subcircuit_queue[qpu_index].append((tape, n_shots))

    def esegui_sottocircuiti(self):
        """esegue tutti i sottocircuiti in coda su ciascuna QPU."""
        for qpu in self.qpus:
            # recupera la lista di (tape, shots) dalla coda per la QPU corrente
            circuits = self.subcircuit_queue[qpu.index]
            if not circuits:
                print(f"Nessun sottocircuito in coda per la QPU {qpu.index}.")
                continue

            print(f"Esecuzione dei sottocircuiti per la QPU {qpu.index}...")
            for i, (tape, n_shots) in enumerate(circuits):

                # non riesco ad assegnare dinamicametne gli shots (quindi quando eseguo il circuito)
                # perciò creo un device per ogni sottocircuito
                dev = qml.device(qpu.tipo, wires = qpu.capacita, shots=n_shots) 

                # esecuzione del tape
                results_list = qml.execute([tape], dev)
                res = results_list[0]

                print(f"Subcircuito {i} eseguito su QPU {qpu.index} con {n_shots} shots.")
                print(f"Risultati: {res}")



    def stampa_coda(self):
        """
        Stamp la coda
        """
        for qpu_index, circuits in self.subcircuit_queue.items():
            print(f"QPU {qpu_index}:")
            if not circuits:
                print(" Non ci sono sottocircuiti assegnati per questa QPU ")
            for i, (tape, n_shots) in enumerate(circuits):
                print(f"  Sottocircuito {i}:")
                print(f"    Shots: {n_shots}")
                print("    Operazioni:")
                for op in tape.operations:
                    print(f"      {op}")
                print("    Osservabili:")
                for ob in tape.observables:
                    print(f"      {ob}")
                print("--------------------------------")

        
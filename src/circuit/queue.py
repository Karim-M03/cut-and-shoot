import random
import pennylane as qml

class CircuitQueue:
    def __init__(self, qpus):

        subcircuit_queue = {}

        for q in qpus:
            subcircuit_queue[q.index] = []

        self.subcircuit_queue = subcircuit_queue
        self.qpus = qpus
        

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

        
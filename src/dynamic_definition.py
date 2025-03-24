from qiskit import QuantumCircuit
from typing import List, Dict, Tuple
import math

def merge_counts(counts: Dict[str, int],
                 merged_qubits: List[int]) -> Dict[str, int]:
    """
    Raggruppa i conteggi di un circuito in base ai qubit specificati in 'merged_qubits'.
    I bit in 'merged_qubits' vengono 'fusi' in un unico bin, mentre gli altri rimangono distinti.

    Esempio:
    counts = {
      '000': 50,
      '001': 30,
      '100': 10,
      '101': 10
    }
    merged_qubits = [0]  # diciamo che vogliamo unire il bit 0
    => i bit string (in ordine q2 q1 q0, se q0 è LSB) vanno
       interpretati e "collassati" su q0. Se q0 è nel merged,
       ciò significa non distinguerlo più: '000' e '001' vanno in un unico bin
       '00-' (dove '-' indica 'merged'), e '100','101' vanno in '10-'.

    Ritorna un dict con chiave = bit string "parziale",
    dove i bit 'merged' sono sostituiti da '-'.
    """

    merged_dict = {}

    for bitstring, ccount in counts.items():
        # bitstring di lunghezza n_qubits (ad es. '0101')
        # costruiamo una "chiave parziale" sostituendo i bit "fusi"
        partial_key = list(bitstring)
        for mq in merged_qubits:
            # mq = indice del bit da fondere
            # ATTENZIONE: a seconda di come Qiskit codifica l'ordine (LSB/MSB)
            # potresti dover invertire l'indice.
            idx = len(bitstring) - 1 - mq
            partial_key[idx] = '-'

        partial_key = ''.join(partial_key)

        if partial_key not in merged_dict:
            merged_dict[partial_key] = 0
        merged_dict[partial_key] += ccount

    return merged_dict


def dynamic_definition(
    list_of_counts: List[Dict[str, int]],
    n_qubits: int,
    active_qubits: int,
    max_recursion: int = 2,
    threshold: float = 0.05
):
    """
    Esempio di funzione che implementa un meccanismo di 'Dynamic Definition':
    - 'list_of_counts': i conteggi di varianti di uno stesso 'sottocircuito' (in un caso realistico
       sarebbero con basi di misura/inizializzazione diverse). Qui, per semplicità, li trattiamo
       come 'dati' e basta.
    - 'n_qubits': numero di qubit totale.
    - 'active_qubits': quanti qubit 'distinguere' in questa fase (gli altri vengono 'fusi').
    - 'max_recursion': quanti step di 'zoom' vogliamo tentare.
    - 'threshold': soglia per decidere se un bin è abbastanza probabile da meritare uno 'zoom' ulteriore.

    L'idea:
     1) Fondiamo i qubit in eccesso in un bin unico.
     2) Calcoliamo la probabilità di ogni 'bin' (somma su list_of_counts).
     3) Se un bin supera la soglia, proviamo a ricorrere 'zoomando' su di esso,
        cioè spostando parte dei qubit 'fusi' a 'attivi'.
    """

    # Se non c'è ricorsione, ci fermiamo e ritorniamo la "vista" con i qubit attivi
    if max_recursion <= 0 or active_qubits >= n_qubits:
        # Ritorniamo semplicemente la fusione base di tutti i 'list_of_counts'
        # in un dizionario globale
        final_merged = {}
        # Merged qubits = tutti tranne i 'active_qubits' (in un meccanismo a scelte 'fisse'
        # potresti passare questa info come arg)
        # qui, semplifichiamo e diciamo che i 'merged_qubits' sono i n_qubits - active_qubits
        # starting from the highest indices. Ovviamente è una semplificazione...
        merged_indices = list(range(n_qubits - active_qubits))

        for cts in list_of_counts:
            partial = merge_counts(cts, merged_indices)
            for pk, val in partial.items():
                final_merged[pk] = final_merged.get(pk, 0) + val

        # Normalizziamo i conteggi
        total_counts = sum(final_merged.values())
        final_probs = {k: v / total_counts for k, v in final_merged.items()}
        return final_probs

    # ALTRIMENTI, proviamo a fare un primo "livello" di fusione,
    # e poi "zoomare" su un bin più probabile
    merged_indices = list(range(n_qubits - active_qubits))
    # uniamo i conteggi di TUTTE le varianti
    big_merged = {}
    for cts in list_of_counts:
        partial = merge_counts(cts, merged_indices)
        for pk, val in partial.items():
            big_merged[pk] = big_merged.get(pk, 0) + val

    total_counts = sum(big_merged.values())
    # Costruiamo la mappa pk -> probabilità
    prob_map = [(k, big_merged[k] / total_counts) for k in big_merged]
    # Ordiniamo discendendo per probabilità
    prob_map.sort(key=lambda x: x[1], reverse=True)

    # Scegliamo i bin 'più probabili' sopra la soglia, e ricorriamo
    # su di essi, con un numero di qubit attivi + 1 (o + N, dipende dalla strategia).
    results = {}
    for pk, pval in prob_map:
        if pval < threshold:
            # sotto la soglia -> non "zoomiamo" più
            results[pk] = pval
        else:
            # simuliamo lo "zoom": passiamo a dynamic_definition con active_qubits + 1
            # e un recursion step in meno
            # *In un caso reale dovremmo rigenerare i circuiti (o filtrare i conteggi)
            #  coerenti col bin pk. Qui ci limitiamo a passare di nuovo la full list_of_counts
            #  perché mancano i meccanismi veri di slicing delle misure. *
            sub_res = dynamic_definition(
                list_of_counts,
                n_qubits,
                active_qubits + 1,
                max_recursion - 1,
                threshold
            )
            # sub_res è un dizionario pk_sub -> prob
            # Ora, i pk_sub corrispondono a un "zoom" più dettagliato
            # Dovremmo "comporli" con pk, ma semplifichiamo:
            for subk, subp in sub_res.items():
                # Attenzione: i subk e pk sono stringhe diverse
                # In un vero design faresti un "merge" più sofisticato delle chiavi
                # Qui facciamo un "concatenate" fittizio
                new_key = f"{pk}|{subk}"
                # E la probabilità va moltiplicata per la frazione pval (approssimazione, in verità)
                # L’idea è che stiamo 'zoomando' dentro pk
                results[new_key] = subp * pval
    # Normalizziamo i results
    ssum = sum(results.values())
    if ssum > 1e-15:
        for k in results.keys():
            results[k] /= ssum
    return results


# ======================= Esempio d'uso =======================

if __name__ == "__main__":
    from qiskit import QuantumCircuit
    # Importa la tua run_quantum_circuits_aer
    from runner_old import run_quantum_circuits_aer

    # Creiamo alcune 'varianti' di un sottocircuito (es. circuiti con misure leggermente diverse).
    qc_base = QuantumCircuit(2, 2)
    qc_base.h(0)
    qc_base.cx(0, 1)
    # Prima variante: misura diretta
    qc1 = qc_base.copy()
    qc1.measure([0, 1], [0, 1])

    # Seconda variante: aggiungiamo un gate su qubit 1 prima di misurare
    qc2 = qc_base.copy()
    qc2.z(1)
    qc2.measure([0, 1], [0, 1])

    # Eseguiamo i circuiti in un colpo solo (QASM)
    circuits_list = [qc1, qc2]
    results_list = run_quantum_circuits_aer(circuits_list,
                                            backend_type="aer_simulator",
                                            shots=1024)
    # results_list è una lista di dict: [counts_qc1, counts_qc2]

    print("\n=== Risultati raw ===")
    for i, cts in enumerate(results_list):
        print(f"Variante {i} ->", cts)

    # Usiamo la "dynamic_definition"
    # Supponiamo di avere 2 qubit totali, e di voler "attivarne" 1 alla volta,
    # con max_recursion = 2.
    dd_probs = dynamic_definition(results_list, n_qubits=2,
                                  active_qubits=1,
                                  max_recursion=2,
                                  threshold=0.1)
    print("\n=== Risultati dynamic definition ===")
    for k, p in dd_probs.items():
        print(f"{k}: {p:.4f}")

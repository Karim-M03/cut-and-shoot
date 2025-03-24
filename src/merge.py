from collections import defaultdict, Counter
from typing import List, Dict, Any

def merge_and_normalize_variant_counts(
    qpu_results_list: List[Dict[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    """
    merges results of subcircuit variants run across multiple QPUs
    each variant is merged by adding raw counts and normalized based on total shots.

    :param qpu_results_list: List of QPU results.
        Each QPU result is a dict: {variant_name: {
            'counts': {bitstring: raw count},
            'active_qubits': [...]
        }}
    :return: Dict of {variant_name: {
        'probabilities': {bitstring: prob},
        'active_qubits': [...]
    }}
    """
    merged_counts = defaultdict(Counter)
    total_shots_per_variant = Counter()
    active_qubits_map = {}

    # merge counts and store active qubits
    for qpu_result in qpu_results_list:
        for variant, data in qpu_result.items():
            counts = data.get('counts', {})
            active_qubits = data.get('active_qubits', [])

            cleaned_counts = {k.strip(): v for k, v in counts.items()}
            merged_counts[variant].update(cleaned_counts)
            total_shots_per_variant[variant] += sum(cleaned_counts.values())

            # only store active qubits once
            if variant not in active_qubits_map:
                active_qubits_map[variant] = active_qubits

    # normalize
    final_results = {}
    for variant, counts in merged_counts.items():
        total_shots = total_shots_per_variant[variant]
        if total_shots == 0:
            continue
        final_results[variant] = {
            'probabilities': {
                bitstring: count / total_shots
                for bitstring, count in counts.items()
            },
            'active_qubits': active_qubits_map.get(variant, [])
        }

    return final_results

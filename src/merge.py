from collections import defaultdict, Counter
from typing import List, Dict, Any


def merge_and_normalize_variant_counts(
    qpu_results_list: List[Dict[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    """
    merges results of subcircuits run across multiple QPUs.
    Eech variant is merged by summing counts and normalizing based on total shots.

    :param qpu_results_list: List of QPU results.
    :return: Dict of {variant_name: {
        'probabilities': {bitstring: prob},
        'active_qubits': [...],
        'initialized_qubits': [...],
        'measured_qubits': [...],
    }}
    """
    merged_counts = defaultdict(Counter)
    total_shots_per_variant = defaultdict(int)
    metadata_per_variant = {}

    # accumulate counts and collect metadata
    for qpu_result in qpu_results_list:
        for variant_name, data in qpu_result.items():
            counts = data.get("counts", {})
            total_shots = data.get("total_shots", sum(counts.values()))
            merged_counts[variant_name].update(counts)
            total_shots_per_variant[variant_name] += total_shots

            # save metadata only once
            if variant_name not in metadata_per_variant:
                metadata_per_variant[variant_name] = {
                    "active_qubits": data.get("active_qubits", []),
                    "initialized_qubits": data.get("initialized_qubits", []),
                    "measured_qubits": data.get("measured_qubits", [])
                }

    # normalize and attach metadata
    final_results = {}
    for variant_name, counts in merged_counts.items():
        total_shots = total_shots_per_variant[variant_name]
        probabilities = {
            bitstring: count / total_shots
            for bitstring, count in sorted(counts.items())
        }

        final_results[variant_name] = {
            "probabilities": probabilities,
            "active_qubits": metadata_per_variant[variant_name]["active_qubits"],
            "initialized_qubits": metadata_per_variant[variant_name]["initialized_qubits"],
            "measured_qubits": metadata_per_variant[variant_name]["measured_qubits"],
        }

    return final_results

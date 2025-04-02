from collections import defaultdict, Counter
from typing import List, Dict, Any


def merge_and_normalize_variant_counts(
    qpu_results_list: List[Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    merges results of subcircuits run across multiple QPUs.
    Each variant is merged by summing counts and normalizing based on total shots.

    :param qpu_results_list: List of QPU results.
    :return: Dict of {sub_id: {
        variant_name: {
            'probabilities': {bitstring: prob}
        }
    }}
    """
    merged_counts = defaultdict(Counter)
    total_shots_per_variant = defaultdict(int)

    # accumulate counts and collect total shots
    for qpu_result in qpu_results_list:
        for variant_name, data in qpu_result.items():
            counts = data.get("counts", {})
            total_shots = data.get("total_shots", sum(counts.values()))
            merged_counts[variant_name].update(counts)
            total_shots_per_variant[variant_name] += total_shots

    # normalize and organize results by sub_id
    final_results = defaultdict(dict)
    for variant_name, counts in merged_counts.items():
        sub_id = variant_name.split("_")[1]
        total_shots = total_shots_per_variant[variant_name]
        probabilities = {
            bitstring: count / total_shots
            for bitstring, count in sorted(counts.items())
        }
        final_results[sub_id][variant_name] = {
            "probabilities": probabilities,
        }

    return final_results

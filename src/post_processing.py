import numpy as np
import itertools
import concurrent.futures
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Union

def vector_from_probabilities(probabilities: Dict[str, float], n: int) -> np.ndarray:
    """
    converts a dictionary of bitstring probabilities into a numpy vector of size 2^n
    each index corresponds to the decimal value of its bitstring key.
    example: for 101 index = 5.
    """
    vec = np.zeros(2**n, dtype=float)
    for bitstring, prob in probabilities.items():
        index = int(bitstring, 2)
        vec[index] = prob
    return vec

def compute_variant_contrib(
    variant_name: str,
    variant_data: Dict[str, Union[List[Any], Dict[str, float]]],
    base_coeff: float
) -> Tuple[np.ndarray, float, int]:
    """
    processes a single variant and it produces:
      - its probability vector
      - its coefficient
      - number of active qubits
    
    the sign of the coefficient depends on the number of x/y measurements (odd = negative)
    """
    x_count = variant_name.count("x")
    y_count = variant_name.count("y")
    sign = +1 if ((x_count + y_count) % 2 == 0) else -1
    eff_coeff = sign * base_coeff

    n_qubits = len(variant_data["active_qubits"])
    prob_dict = variant_data["probabilities"]
    vec = vector_from_probabilities(prob_dict, n_qubits)

    return vec, eff_coeff, n_qubits

def fd_reconstruct_variant_dict(
    all_results: Dict[str, Dict[str, Union[List[Any], Dict[str, float]]]],
    base_coeff: float
) -> np.ndarray:
    """
    performs full definition by:
      - grouping all variant results by subcircuit
      - computing kronecker products across all subcircuit combinations
      - summing weighted contributions to obtain global distribution
    """
    # group variants by subcircuit index
    grouped = defaultdict(dict)
    for variant_name, data in all_results.items():
        if variant_name.startswith("sub_"):
            parts = variant_name.split("_")
            try:
                sub_idx = int(parts[1])  # extract subcircuit index from sub_x
            except ValueError:
                sub_idx = variant_name  # fallback if not numeric
        else:
            sub_idx = variant_name
        grouped[sub_idx][variant_name] = data

    # build variant data for each subcircuit
    sub_variant_lists = []
    for sub_idx in sorted(grouped.keys()):
        variants_dict = grouped[sub_idx]
        local_list = []
        for var_name, var_data in variants_dict.items():
            vec, eff_coeff, n_q = compute_variant_contrib(var_name, var_data, base_coeff)
            local_list.append((vec, eff_coeff, n_q))
        sub_variant_lists.append(local_list)

    # all variants in each subcircuit have the same number of active qubits
    total_active = sum(v_list[0][2] for v_list in sub_variant_lists)
    final_dim = 2 ** total_active
    global_vector = np.zeros(final_dim, dtype=float)

    def combination_contribution(combo: Tuple[Tuple[np.ndarray, float, int], ...]) -> np.ndarray:
        """
        given a tuple of (vec, coeff, n_qubits) for each subcircuit,
        compute kronecker product and apply total coefficient
        """
        for (vec, coeff, n) in combo:
            if np.allclose(vec, 0):
                dims = [len(v) for v, _, _ in combo]
                return np.zeros(np.prod(dims), dtype=float)

        total_coeff = np.prod([coeff for _, coeff, _ in combo])
        kron_prod = combo[0][0]
        for (vec, _, _) in combo[1:]:
            kron_prod = np.kron(kron_prod, vec)

        return total_coeff * kron_prod

    # generate all combinations of variants
    all_combinations = list(itertools.product(*sub_variant_lists))

    # compute contributions in parallel
    partial_contributions = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_list = [executor.submit(combination_contribution, combo) for combo in all_combinations]
        for fut in concurrent.futures.as_completed(future_list):
            partial_contributions.append(fut.result())

    # sum all contributions
    global_vector = np.sum(partial_contributions, axis=0)

    # normalize the final result if not zero
    total_prob = np.sum(global_vector)
    if abs(total_prob) > 1e-15:
        global_vector /= total_prob

    return global_vector

def dd_reconstruct_variant_dict(
    all_results: Dict[str, Dict[str, Union[List[Any], Dict[str, float]]]],
    base_coeff: float
) -> np.ndarray:
    """
    performs dynamic definition reconstruction by:
      - computing the global vector entry by entry
      - using only the relevant subvector entries and their combinations
    """
    # group variants by subcircuit
    grouped = defaultdict(dict)
    for variant_name, data in all_results.items():
        if variant_name.startswith("sub_"):
            parts = variant_name.split("_")
            try:
                sub_idx = int(parts[1])
            except ValueError:
                sub_idx = variant_name
        else:
            sub_idx = variant_name
        grouped[sub_idx][variant_name] = data

    subcircuit_keys = sorted(grouped.keys())
    sub_vectors = []

    # for each subcircuit, build index_to_value + coefficient mapping
    for sub_idx in subcircuit_keys:
        variant_entries = defaultdict(list)
        for variant_name, data in grouped[sub_idx].items():
            vec, eff_coeff, n = compute_variant_contrib(variant_name, data, base_coeff)
            for idx, value in enumerate(vec):
                if value != 0:
                    variant_entries[idx].append((value, eff_coeff))
        n_qubits = len(grouped[sub_idx][next(iter(grouped[sub_idx]))]["active_qubits"])
        sub_vectors.append((n_qubits, variant_entries))

    total_active = sum(n for n, _ in sub_vectors)
    final_dim = 2**total_active
    global_vector = np.zeros(final_dim)

    # reconstruct each index individually
    for global_idx in range(final_dim):
        bin_str = format(global_idx, f"0{total_active}b")  # binary string for this index
        offset = 0
        total_contrib = 0.0
        valid = True
        sub_values = []

        # extract local indices for each subcircuit
        for n, entries in sub_vectors:
            local_bits = bin_str[offset:offset+n]
            local_idx = int(local_bits, 2)
            offset += n
            if local_idx not in entries:
                valid = False
                break
            sub_values.append(entries[local_idx])

        if not valid:
            continue

        # compute all possible combinations of values from each subcircuit
        for combo in itertools.product(*sub_values):
            coeffs = [val * eff for val, eff in combo]
            total_contrib += np.prod(coeffs)

        global_vector[global_idx] = total_contrib

    # normalize the result
    total = np.sum(global_vector)
    if abs(total) > 1e-15:
        global_vector /= total

    return global_vector

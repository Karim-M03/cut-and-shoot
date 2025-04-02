from typing import Any, Dict, List

def format_data(
    quantum_subcircuits: Dict[int, Dict[str, Any]],
    variants_results: Dict[str, Dict[str, Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    builds the subcircuits_data list in a format to facilitate the reconstruction,
    """
    subcircuits_data: List[Dict[str, Any]] = []

    # scan all subcircuit IDs and their associated variant objects
    for sub_id, variant_dict in quantum_subcircuits.items():
        str_sub_id = str(sub_id)  # keys in variants_results are stringified sub_ids

        for variant_name, variant_obj in variant_dict.items():
            # check if this variant has results available
            if str_sub_id not in variants_results:
                continue
            if variant_name not in variants_results[str_sub_id]:
                continue

            output_distribution = variants_results[str_sub_id][variant_name]["probabilities"]

            # for each cut-out edge treat it as an upstream role (with measurement nodes)
            for out_info in variant_obj.cuts_info.get("out", []):
                cut_id = out_info["cut_id"]
                edge = out_info["edge"]  # (source_vertex, target_vertex)

                subcircuits_data.append({
                    "subcircuit_id": sub_id,
                    "cut_id": cut_id,
                    "edge": edge,
                    "role": "upstream",
                    "measurement_bases": variant_obj.measured_info,  # use local qubit indices
                    "output_distribution": output_distribution,
                    "bitstring_mapping": variant_obj.qbit_map
                })

            # for each cut-in edge treat it as a downstream role (with initialized qubits)
            for in_info in variant_obj.cuts_info.get("in", []):
                cut_id = in_info["cut_id"]
                edge = in_info["edge"]

                subcircuits_data.append({
                    "subcircuit_id": sub_id,
                    "cut_id": cut_id,
                    "edge": edge,
                    "role": "downstream",
                    "init_states": variant_obj.initialized_info,  # use local qubit indices
                    "output_distribution": output_distribution,
                    "bitstring_mapping": variant_obj.qbit_map
                })

    return subcircuits_data

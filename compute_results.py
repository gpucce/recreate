import re
import ast
import pandas as pd
import os
import json
from collections import defaultdict, Counter

def load_partial_entities(raw: str, key: str = "named_entities"):
    s = raw.strip()

    # If it's a Python-style quoted string with literal \n escapes, decode it first
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        try:
            s = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            s = s[1:-1].replace("\\n", "\n")

    # Strip the markdown fence
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)

    # Jump to the start of the array
    start = s.index("[", s.index(key)) + 1

    decoder = json.JSONDecoder()
    items, i = [], start
    while True:
        while i < len(s) and s[i] in " \t\r\n,":   # skip whitespace and commas
            i += 1
        if i >= len(s) or s[i] == "]":             # clean end of array
            break
        try:
            obj, i = decoder.raw_decode(s, i)      # parse one complete value
        except json.JSONDecodeError:
            break                                   # truncated or malformed: stop here
        items.append(obj)

    # make sure that all the items are dicts with keys "entity" and "type"
    clean = []
    for elem in items:
        if isinstance(elem, dict) and "entity" in elem and "type" in elem:
            clean.append(elem)
        else:
            pass
            # print(f"-- Skipping malformed entity in cleaning: {elem}")
    return {key: clean}


def main(args):
    print(f"Processing files in folder: {args.target_folder}")
    _, _, dataset_name, model_name = args.target_folder.split("/")
    print(f"Model name inferred from folder: {model_name}")
    print(f"Dataset name inferred from folder: {dataset_name}")

    files = sorted([f for f in os.listdir(args.target_folder) if f.endswith(".json")])
    try:
        files.remove("original.json") # Remove "original.json" if it exists
        files = ["original.json"] + files
    except ValueError:
        pass
    try:
        files.remove("original.debug.json")  # Remove "original.debug.json" if it exists
        files = ["original.debug.json"] + files
    except ValueError:
        pass

    print(f"Found {len(files)} JSON files.")
    for file in files:
        print(f"- {file}")

    iter_dict = {f.replace(".json", "").replace(".debug", ""): json.load(open(os.path.join(args.target_folder, f), "r")) for f in files}
    # print(f"Iter dict keys: {list(iter_dict.keys())}")

    # sanitize iter dict before computing counts and document tracking: TODO double check
    print("Sanitizing iter_dict to ensure all samples have 'named_entities' and are well-formed...")
    clean_iter_dict = {}
    for iter_k, iter_v in iter_dict.items():
        clean_iter_v = []
        for i, sample in enumerate(iter_v):
            if "named_entities" not in sample:
                sample = load_partial_entities(str(sample["raw_output"]))
                if "named_entities" not in sample:
                    # print(f"Skipping element without 'named_entities' at iteration {iter_k}, sample {i}") #: {sample}")
                    continue
            named_entities = sample["named_entities"]
            clean_named_entities = []
            for item in named_entities:
                if isinstance(item, dict) and "entity" in item and "type" in item:
                    clean_named_entities.append(item)
                else:
                    # print(f"-- Skipping malformed entity at iteration {iter_k}, sample {i}: {item}")
                    pass
            sample["named_entities"] = clean_named_entities
            clean_iter_v.append(sample)
        clean_iter_dict[iter_k] = clean_iter_v
    iter_dict = clean_iter_dict

    # get counts of entity/type across iterations and track document ids for each entity/type across iterations
    counts, document_tracking, malformed_samples = get_count_and_document_trackers(iter_dict, level="type")
    print(f"-- Found {len(malformed_samples)} malformed samples across iterations")

    count_entities, entity_type_mapper = get_count_and_document_trackers_entity_level(iter_dict)

    # merge count and document tracking into a single dataframe with columns: iterations (one for each iter), entity/type, count, document_ids
    merged_data = []
    for iter_k, entity_counts in counts.items():
        for entity, count in entity_counts.items():
            document_ids = list(document_tracking[iter_k][entity])
            merged_data.append({
                "iteration": iter_k,
                "entity": entity,
                "count": count,
                "document_ids": document_ids
            })
    df_type = pd.DataFrame(merged_data)

    merged_data_entity_level = []
    for iter_k, entity_counts in count_entities.items():
        for entity, count in entity_counts.items():
            for elem, cnt in count.items():
                merged_data_entity_level.append({
                    "iteration": iter_k,
                    "entity": elem,
                    "count": cnt,
                    "type": entity_type_mapper[elem],
                    "document_ids": entity
                }) 
    df_entity = pd.DataFrame(merged_data_entity_level)

    outdir = os.path.join("outputs", "textonly_dataframes", dataset_name, model_name)
    os.makedirs(outdir, exist_ok=True)

    print(f"Filtered DataFrame saved to {outdir}")

    df_type.to_json(os.path.join(outdir, f"counts_type.json"), orient="records", indent=4)
    df_entity.to_json(os.path.join(outdir, f"counts_entity.json"), orient="records", indent=4)

    # filter df to include rows where id 0 appears in document_ids for each iteration
    # df_filtered = df[df["document_ids"].apply(lambda ids: 0 in ids)]


def get_count_and_document_trackers_entity_level(iter_dict: dict):
    iter_counters = defaultdict(lambda: defaultdict(int))
    entity_type_mapper = {}

    # for each iteration, for each document create a counter of entities
    for iter_k, iter_v in iter_dict.items():
        for i, sample in enumerate(iter_v):
            sample_counter = Counter()
            if "raw_output" in sample:
                parsing = sample["raw_output"]
                sample = load_partial_entities(parsing)
            sample_counter.update(entity["entity"].lower() for entity in sample["named_entities"])
            iter_counters[iter_k][i] = sample_counter

            for entity in sample["named_entities"]:
                entity_type_mapper[entity["entity"].lower()] = entity["type"].lower()

    return iter_counters, entity_type_mapper


def get_count_and_document_trackers(iter_dict: dict, level: str = "type"):
    malformed_samples = []
    counts = defaultdict(lambda: defaultdict(int))
    document_tracking = defaultdict(lambda: defaultdict(set))  # Track document IDs for each entity/type across iterations
    for iter_k, iter_v in iter_dict.items():
        print(f"---- Processing iteration: {iter_k} -----\n")
        for i, sample in enumerate(iter_v):
            # assert "named_entities" in sample, f"'named_entities' key not found in sample: {sample}"
            if "raw_output" in sample:
                # TODO: here we forcefully parse the raw output to get the named_entities, but this is a hacky solution. We should fix the data generation process to ensure that named_entities are always present in the sample.
                # print("Hit a malformed sample, attempting to parse 'raw_output' to extract 'named_entities'.")
                parsing = sample["raw_output"]
                sample = load_partial_entities(parsing)
                # keep track of the malformed sample for debugging
                malformed_samples.append((iter_k, i))
                print(f"Iteration {iter_k}, Sample {i}: Malformed sample detected. Parsed 'raw_output' to extract 'named_entities'.")
                continue
            for entity in sample["named_entities"]:
                if level not in entity:
                    print(f"Warning: '{level}' key not found in entity: {entity} at iteration {iter_k}, sample {i}. Skipping this entity.")
                    continue    # TODO FIXME
                if type(entity) == str:
                    print(f"-- Warning: Entity is a string instead of a dict at iteration {iter_k}, sample {i}. Entity: {entity}. Skipping this entity.")
                    continue    # TODO FIXME
        
                # print(f"Iteration {iter_k}, Sample {i}: available keys in entity: {list(entity.keys())}, using '{level}' key for counting. Entity: {entity}")
                counts[iter_k][entity[level].lower()] += 1
                document_tracking[iter_k][entity[level].lower()].add(i)  # Track document ID (index) for each entity type
    return counts, document_tracking, malformed_samples


def get_vocab(iter_dict: dict, level: str = "type") -> set:
    vocab = set()
    for iter_k, iter_v in iter_dict.items():
        # iterate over all the samples in the iteration/step
        for sample in iter_v:
            assert "named_entities" in sample, f"'named_entities' key not found in sample: {sample}"
            for entity in sample["named_entities"]:
                vocab.add(entity[level])
    print(f"Vocabulary of '{level}':\n- Size: {len(vocab)}") #\n- Content: {vocab}")
    return vocab


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-folder", type=str, default="outputs/textonly_describe/erc-grants/qwen_qwen3.8_27b", help="Path to the target folder")
    args = parser.parse_args()

    main(args)
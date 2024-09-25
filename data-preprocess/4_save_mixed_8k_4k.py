import json
import random
from utils import save_json
import fire

def get_sampled_texts(document_texts, sample_len, random_seed=42):
    random.seed(random_seed)
    random.shuffle(document_texts)
    current_len = 0
    sampled_texts = []

    for text in document_texts:
        if current_len + len(text) <= sample_len:
            sampled_texts.append(text)
            current_len += len(text)
        else:
            # If adding the full text exceeds sample_len, truncate the text
            remaining_len = sample_len - current_len
            if remaining_len > 0:
                sampled_texts.append(text[:remaining_len])
            break

    return sampled_texts

def save_mixed(rel_data_path, unrel_data_path, mixed_data_path, random_seed=42):
    random.seed(random_seed)
    
    mixed_data_list = []

    with open(rel_data_path, "r") as rel, open(unrel_data_path, "r") as unrel:
        rel_datasets = json.load(rel)
        unrel_datasets = json.load(unrel)
        total_len = len(rel_datasets)

        for i in range(total_len):
            rel_dataset = rel_datasets[i]
            unrel_dataset = unrel_datasets[i]

            rel_document = rel_dataset["document_text"]
            unrel_document = unrel_dataset["document_text"]

            # Calculate total length of all strings in the list
            total_rel_len = sum(len(text) for text in rel_document)
            total_unrel_len = sum(len(text) for text in unrel_document)

            # Half of the total length
            sample_rel_len = total_rel_len // 2
            sample_unrel_len = total_unrel_len // 2

            # Remove gold document from rel_document
            gold_document = rel_dataset["annotations"]["long_answer"]
            if gold_document in rel_document:
                rel_document.remove(gold_document)

            # Get sampled texts
            sampled_rel = get_sampled_texts(rel_document, sample_rel_len, random_seed)
            sampled_unrel = get_sampled_texts(unrel_document, sample_unrel_len, random_seed)

            related_information = list(range(len(sampled_rel)))

            sampled_rel.extend(sampled_unrel)


            formatted_mixed_data_dict = {
                "title": rel_dataset["title"],
                "document_text": sampled_rel,
                "related_information": related_information,
                "question_text": rel_dataset["question_text"],
                "annotations": rel_dataset["annotations"],
                "document_url": {
                    "rel": rel_dataset["document_url"],
                    "unrel": unrel_dataset["document_url"]
                },
                "example_id": f"{i}_mix"
            }
            mixed_data_list.append(formatted_mixed_data_dict)

        save_json(mixed_data_path, mixed_data_list)

if __name__ == "__main__":
    fire.Fire(save_mixed)

    # For 8k_mixed, run the file with the folllowing command
    # python 4_save_mixed_8k_4k.py \
    #   --rel_data_path "{rel_8k_data_path}" \
    #   --unrel_data_path "{unrel_8k_data_path}" \
    #   --mixed_data_path "{path_to_store_8k_mixed_data}"

    """
    우리의 경우, 8k
    python 4_save_mixed_8k_4k.py \
        --rel_data_path "/data/yjoonjang/datasets/long_context/8k_rel.json" \
        --unrel_data_path "/data/yjoonjang/datasets/long_context/8k_unrel.json" \
        --mixed_data_path "{저장할 8k_mixed path}"
    """

    # For 4k_mixed, run the file with the folllowing command
    # python 4_save_mixed_8k_4k.py \
    #   --rel_data_path "{rel_4k_data_path}" \
    #   --unrel_data_path "{unrel_4k_data_path}" \
    #   --mixed_data_path "{path_to_store_4k_mixed_data}"

    """
    우리의 경우, 4k
    python 4_save_mixed_8k_4k.py \
        --rel_data_path "/data/yjoonjang/datasets/long_context/4k_rel.json" \
        --unrel_data_path "/data/yjoonjang/datasets/long_context/4k_unrel.json" \
        --mixed_data_path "{저장할 4k_mixed path}"
    """
import json
import random
from utils import save_json

import torch
import torch.nn as nn
from tqdm import tqdm
import fire

import importlib

save_unrel_16k = importlib.import_module("1_save_unrel_16k")

def get_excluded_indices(document_texts, long_answer):
    indices = list(range(0, len(document_texts)))
    excluded_indices = [index for index in indices if document_texts[index] != long_answer]

    if len(excluded_indices) == len(indices):
        print("No index found for long_answer !!")
        return

    return excluded_indices

def get_8k_4k_dataset(dataset, cos, is_rel=True):
    processed_data_8k = []
    processed_data_4k = []

    for index, data in tqdm(enumerate(dataset)):
        gold_text = data["annotations"]["long_answer"]
        question_text = data["question_text"]
        question_embedding = torch.tensor(save_unrel_16k.get_embedding(question_text)).unsqueeze(0)

        document_texts = data["document_text"]
        document_text_embeddings = []

        paragraph_len = sum(len(text) for text in document_texts)
        for document_text in document_texts:
            document_text_embedding = torch.tensor(save_unrel_16k.get_embedding(document_text))
            document_text_embeddings.append(document_text_embedding)

        document_text_embeddings = torch.stack(document_text_embeddings)

        with torch.no_grad():
            similarity = cos(question_embedding, document_text_embeddings)

        sim_len_pairs = list(zip(similarity.tolist(), document_texts))

        if is_rel:
            sim_len_pairs.sort(key=lambda x: x[0], reverse=True)
        else:
            sim_len_pairs.sort(key=lambda x: x[0])

        selected_texts_8k = [gold_text] if is_rel else []
        selected_texts_4k = [gold_text] if is_rel else []

        selected_len_8k = len(gold_text) if is_rel else 0
        selected_len_4k = len(gold_text) if is_rel else 0
        half_paragraph_len = paragraph_len * 0.5
        quarter_paragraph_len = paragraph_len * 0.25

        for sim, text in sim_len_pairs:
            if text != gold_text:
                text_len = len(text)
                if selected_len_8k + text_len <= half_paragraph_len:
                    selected_texts_8k.append(text)
                    selected_len_8k += text_len
                else:
                    remaining_len = half_paragraph_len - selected_len_8k
                    if remaining_len > 0:
                        selected_texts_8k.append(text[:int(remaining_len)])
                    break

        for sim, text in sim_len_pairs:
            if text != gold_text:
                text_len = len(text)
                if selected_len_4k + text_len <= quarter_paragraph_len:
                    selected_texts_4k.append(text)
                    selected_len_4k += text_len
                else:
                    remaining_len = quarter_paragraph_len - selected_len_4k
                    if remaining_len > 0:
                        selected_texts_4k.append(text[:int(remaining_len)])
                    break

        random.shuffle(selected_texts_8k)
        random.shuffle(selected_texts_4k)

        dataset_dict_8k = {
            "title": data["title"],
            "document_text": selected_texts_8k,
            "question_text": data["question_text"],
            "annotations": data["annotations"],
            "document_url": data["document_url"],
            "example_id": f"{index}_{'rel' if is_rel else 'unrel'}"
        }

        dataset_dict_4k = {
            "title": data["title"],
            "document_text": selected_texts_4k,
            "question_text": data["question_text"],
            "annotations": data["annotations"],
            "document_url": data["document_url"],
            "example_id": f"{index}_{'rel' if is_rel else 'unrel'}"
        }

        if is_rel:
            indices_list_8k = get_excluded_indices(selected_texts_8k, gold_text)
            indices_list_4k = get_excluded_indices(selected_texts_4k, gold_text)
            dataset_dict_8k["related_information"] = indices_list_8k
            dataset_dict_4k["related_information"] = indices_list_4k

        processed_data_8k.append(dataset_dict_8k)
        processed_data_4k.append(dataset_dict_4k)

    return processed_data_8k, processed_data_4k

def save_8k_4k(
        rel_data_16k_path,
        unrel_data_16k_path,
        rel_data_8k_path,
        unrel_data_8k_path,
        rel_data_4k_path,
        unrel_data_4k_path,
        random_seed=42
    ):
   
    random.seed(random_seed)
    cos = nn.CosineSimilarity(dim=-1)

    with open(rel_data_16k_path, 'r') as rel_16k, open(unrel_data_16k_path, 'r') as unrel_16k:
        rel_16k_datasets = json.load(rel_16k)
        unrel_16k_datasets = json.load(unrel_16k)

        rel_datasets_8k, rel_datasets_4k = get_8k_4k_dataset(rel_16k_datasets, cos, is_rel=True)
        unrel_datasets_8k, unrel_datasets_4k = get_8k_4k_dataset(unrel_16k_datasets, cos, is_rel=False)

        save_json(rel_data_8k_path, rel_datasets_8k)
        save_json(rel_data_4k_path, rel_datasets_4k)
        save_json(unrel_data_8k_path, unrel_datasets_8k)
        save_json(unrel_data_4k_path, unrel_datasets_4k)


if __name__ == "__main__":
    # fire.Fire(save_8k_4k)

    # run the file with the folllowing command
    # python 3_save_rel_unrel_8k_4k.py \
    #   --rel_data_16k_path "/data/yjoonjang/datasets/long_context/16k_rel.json" \
    #   --unrel_data_16k_path "/data/yjoonjang/datasets/long_context/16k_unrel.json" \
    #   --rel_data_8k_path "rel_8k_path" \
    #   --unrel_data_8k_path "unrel_8k_path" \
    #   --rel_data_4k_path "rel_4k_path" \
    #   --unrel_data_4k_path "unrel_4k_path"

    """
    우리의 경우, 
    python 3_save_rel_unrel_8k_4k.py \
        --rel_data_16k_path "/data/yjoonjang/datasets/long_context/16k_rel.json" \
        --unrel_data_16k_path "/data/yjoonjang/datasets/long_context/16k_unrel.json" \
        --rel_data_8k_path "{저장할 8k_rel path}" \
        --unrel_data_8k_path "{저장할 8k_unrel path}" \
        --rel_data_4k_path "{저장할 4k_rel path}" \
        --unrel_data_4k_path "{저장할 4k_unrel path}"
    """
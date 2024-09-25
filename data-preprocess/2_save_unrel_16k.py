import json
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import os
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
from utils import save_json
import fire

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
client = OpenAI(api_key=OPENAI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("long-context-llm")


def get_embedding(text, model="text-embedding-3-large"):
    # text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model).data
    response = response[0].embedding
    return response

def store_vector(vectors: list, namespace: str):
    index.upsert(vectors=vectors, namespace=namespace)

def format_to_vector_dict(vector_id: str, values: list, metadata: dict):
    return {
        "id": vector_id,
        "values": values,
        "metadata": metadata,
    }

def store_vector_to_pinecone(document_texts, namespace):
    vectors = []
    for i in range(len(document_texts)):
        document = "\n".join(document_texts[i])
        values = get_embedding(document)
        vector_id = f"{i}" # id should be str
        metadata = {
            "rel_document_text_index": i
        }
        vector_dict = format_to_vector_dict(vector_id, values, metadata)
        vectors.append(vector_dict)
    store_vector(vectors, namespace)

def get_unrelated_data(vector: list, topk: int, namespace: str):
    unrelated_data = index.query(namespace=namespace, vector=vector, top_k=topk)
    unrelated_data_index = int(unrelated_data["matches"][-1]["id"])
    return unrelated_data_index

def save_unrel(rel_data_path, unrel_data_path, save_to_vectordb):
    with open(rel_data_path, 'r') as rel:
        rel_datas = json.load(rel)
        rel_document_texts = []
        unrel_datas = []
        topk = 100

        if save_to_vectordb:
            for rel_data in tqdm(rel_datas, desc="Saving to vectorDB.."):
                rel_document_texts.append(rel_data["document_text"])
                store_vector_to_pinecone(rel_document_texts, "documents")
            print("Saved to vectorDB successfully!")

        for index, rel_data in tqdm(enumerate(rel_datas), desc="Retrieving unrelated documents.."):
            question_text = rel_data["question_text"]
            
            # retrieve unsimilar document
            question_text_embedding = get_embedding(question_text)
            unrelated_data_index = get_unrelated_data(question_text_embedding, topk=topk, namespace="documents")
            unrelated_document = rel_datas[unrelated_data_index]["document_text"]
            unrelated_document_url = rel_datas[unrelated_data_index]["document_url"]
            unrel_data_dict = {
                "title": rel_data["title"],
                "document_text": unrelated_document,
                "question_text": question_text,
                "annotations": rel_data["annotations"],
                "document_url": unrelated_document_url,
                "example_id": f"{index}_unrel"
            }
            unrel_datas.append(unrel_data_dict)

        save_json(unrel_data_path, unrel_datas)


if __name__ == "__main__":
    fire.Fire(save_unrel)

    # 1. if only retireval needed, run the file with the folllowing command
    # python 1_save_unrel.py \
    #   --rel_data_16k_path "{rel_data_path}" \
    #   --unrel_data_16k_path "{unrel_data_path}" \
    #   --save_to_vectordb=False
    
    # 2. if storing to vectorDB and retrieval needed, run the file with the folllowing command
    # python 1_save_unrel.py \
    #   --rel_data_16k_path "{rel_data_path}" \
    #   --unrel_data_16k_path "{unrel_data_path}" \
    #   --save_to_vectordb=True
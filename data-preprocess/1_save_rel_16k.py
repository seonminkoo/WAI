import json
from html.parser import HTMLParser
from utils import save_json
import argparse

source_file = "/data/koo/datasets/long_context/v1.0-simplified_simplified-nq-train.jsonl"


class TextFromPParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.recording = False
        self.data = []
        self.p_tag_open = False
        self.inside_table = False  # <table> 태그 내부에 있는지 확인하기 위한 플래그

    def handle_starttag(self, tag, attrs):
        if tag.lower() == 'table':
            self.inside_table = True  # <table> 태그 시작
        if tag.lower() == 'p' and not self.inside_table:
            self.recording = True
            self.p_tag_open = True

    def handle_endtag(self, tag):
        if tag.lower() == 'table':
            self.inside_table = False  # <table> 태그 종료
        if tag.lower() == 'p':
            self.recording = False
            self.p_tag_open = False

    def handle_data(self, data):
        if self.recording and self.p_tag_open and not self.inside_table:
            self.data.append(data)


def extract_text_from_p(html_content):
    parser = TextFromPParser()
    parser.feed(html_content)
    return parser.data

# extract title without " - wikipedia"
def extract_title(text):
    hyphen_position = text.find(' - wikipedia')
    if hyphen_position != -1:
        title = text[:hyphen_position]
    else:
        hyphen_position = text.find(' - Wikipedia')
        title = text[:hyphen_position]
    return title

def get_excluded_indices(document_texts, long_answer):
    indices = list(range(0, len(document_texts)))
    excluded_indices = [index for index in indices if document_texts[index] != long_answer]

    if len(excluded_indices) == len(indices):
        print("No index found for long_answer !!")
        return

    return excluded_indices

import json 
import random

def save_long_rel(long_datasets, output_file, random_seed=42):
    long_picked_datasets = []

    # random sampling
    random.seed(random_seed)

    for index, dataset in enumerate(long_datasets):
        document_text = dataset["document_text"]

        title = extract_title(document_text)
        question_text = dataset["question_text"]
        extracted_document_texts = extract_text_from_p(document_text)

        # filter
        banned_text_list = [" ", " . ", "<", "  "]
        extracted_document_texts = [text for text in extracted_document_texts if (text not in banned_text_list) and (len(text) >= 30)] # filter unimportant document_text
        extracted_document_texts = list(set(extracted_document_texts)) # deduplicate

        document_text_tokens = document_text.split()  # tokenize document_text

        long_answer_info = dataset["annotations"][0]["long_answer"]
        short_answers_info = dataset["annotations"][0]["short_answers"]

        short_answers = []
        long_answer = " ".join(document_text_tokens[long_answer_info['start_token']:long_answer_info['end_token']])
        long_answer = extract_text_from_p(long_answer)[0] # delete P tag from long_answer
        for short_answer_info in short_answers_info:
            short_answer = document_text_tokens[short_answer_info['start_token']:short_answer_info['end_token']]
            short_answers.append(" ".join(short_answer))

        indices_list = get_excluded_indices(extracted_document_texts, long_answer)
        if len(indices_list) > 7:
            formatted_data_dict = {
                "title": title,
                "document_text": extracted_document_texts,
                "related_information": indices_list,
                "question_text": question_text,
                "annotations": {
                    "yes_no_answer": "NONE",
                    "long_answer": long_answer,
                    "short_answers": short_answers
                },
                "document_url": dataset["document_url"],
                "example_id": f"{index}_rel"
            }
            long_picked_datasets.append(formatted_data_dict)

    formatted_datasets = random.sample(long_picked_datasets, 100)
    save_json(output_file, formatted_datasets)

if __name__ == "__main__":
    # command
    """
    python 1_save_rel_16k.py --output_file "your-rel-16k-data-path"
    """
    
    parser = argparse.ArgumentParser(description="Process long datasets and save to output file.")
    parser.add_argument('--source_file', type=str, required=True, help='Path to the source file containing datasets')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output file to save processed datasets')
    args = parser.parse_args()

    long_datasets = []

    with open(args.source_file, "r") as file:
        for line in file:
            dataset = json.loads(line)
            document_text = dataset["document_text"]
            document_text_tokens = document_text.split()
            annotations = dataset["annotations"]
            long_answer = " ".join(document_text_tokens[annotations[0]["long_answer"]["start_token"]:annotations[0]["long_answer"]['end_token']])

            # extract 15.9k < len(document_text) <= 16k & exists long_answer & exists short_answers & yes_no_answer == "NONE" & long_answer[:3] == "<P>"
            if ((15900 < len(document_text)) and (len(document_text) <= 16000)) and (annotations[0]["long_answer"]) and (len(annotations[0]["short_answers"]) >= 1) and (annotations[0]["yes_no_answer"] == "NONE") and (long_answer[:3] == "<P>"):
                long_datasets.append(dataset)
    
    save_long_rel(long_datasets=long_datasets, output_file=args.output_file)
  


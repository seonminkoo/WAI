import json

def save_json(file_name, dataset):
    with open(file_name, "w") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=4)
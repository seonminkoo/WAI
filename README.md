This is the official github repository for the paper: [_Where am I?_ Large Language Models Wandering between Semantics and Structures in Long Contexts](https://openreview.net/forum?id=gY1eSVMx2E)

*Seonmin Koo(\*), Jinsung Kim(\*), YoungJoon Jang, Chanjun Park, Heuiseok Lim* 

🏫 [NLP & AI Lab](https://blpkorea.cafe24.com/wp/), Korea University

---
### 🔥 News
- September 20, 2024: Accpeted main paper at EMNLP 2024

### 🛠️ Installation
```bash
$ git clone https://github.com/seonminkoo/WAI.git
```

```bash
# python_requires >=3.9
$ cd WAI
$ pip install -r requirements.txt 
```

### 🚀 Usage

#### Surrounding Context Construction

By following the steps below, you can generate a dataset that encompasses various semantic and structural compositions of the context.

1. Add your `OPENAI_API_KEY` and `PINECONE_API_KEY` to the `.env` file by referring to the `.env.template`.

2. Save 16k related data

    ```bash
    python3 data-preprocess/1_save_rel_16k.py \
      --output_file "./datasets/your-rel-16k-data-path"
    ```

3. Save 16k unrel data

    - If both storing to vectorDB and retrieval needed, run the file with the folllowing command
      ```bash
      python3 data-preprocess/2_save_unrel_16k.py \
        --rel_data_16k_path "./datasets/your-rel-16k-data-path" \
        --unrel_data_16k_path "./datasets/your-unrel-16k-data-path" \
        --save_to_vectordb=True
      ```

    - If only retireval needed, run the file with the folllowing command
      ```bash
      python3 data-preprocess/2_save_unrel_16k.py \
        --rel_data_16k_path "./datasets/your-rel-16k-data-path" \
        --unrel_data_16k_path "./datasets/your-unrel-16k-data-path" \
        --save_to_vectordb=False
      ```

4. Save 16k mixed data

    ```bash
    python3 data-preprocess/3_save_mixed_16k.py \
      --rel_data_path "./datasets/your-rel-16k-data-path" \
      --unrel_data_path "./datasets/your-unrel-16k-data-path" \
      --mixed_data_path "./datasets/your-mixed-16k-data-path"
    ```

5. Save (8k, 4k) (related, unrelated) data

    ```bash
    python3 data-preprocess/4_save_rel_unrel_8k_4k.py \
      --rel_data_16k_path "./datasets/your-rel-16k-data-path" \
      --unrel_data_16k_path "./datasets/your-unrel-16k-data-path" \
      --rel_8k_data_path "./datasets/your-rel-8k-data-path" \
      --unrel_8k_data_path "./datasets/your-unrel-8k-data-path" \
      --rel_4k_data_path "./datasets/your-rel-4k-data-path" \
      --unrel_4k_data_path "./datasets/your-unrel-4k-data-path"
    ```

6. Save 8k, 4k mixed data

    - For 8k mixed data, run the file with the folllowing command

      ```bash
      python3 data-preprocess/5_save_mixed_8k_4k.py \
        --rel_data_path "./datasets/your-rel-8k-data-path" \
        --unrel_data_path "./datasets/your-unrel-8k-data-path" \
        --mixed_data_path "./datasets/your-mixed-8k-data-path"
      ```

    - For 4k mixed data, run the file with the folllowing command

      ```bash
      python3 data-preprocess/5_save_mixed_8k_4k.py \
        --rel_data_path "./datasets/your-rel-4k-data-path" \
        --unrel_data_path "./datasets/your-unrel-4k-data-path" \
        --mixed_data_path "./datasets/your-mixed-4k-data-path"
      ```

#### Verify the task alignment of LLMs

You can execute the generated dataset by providing it to the parameters below. By modifying the shell parameters, you can address various semantic and structural compositions of the context. An API key is required to run the desired model; for instance, to execute the ChatGPT model, an OpenAI API key is necessary.
```bash
$ sh test.sh
```

```bash
## test.sh
MODEL=gpt #gpt, claude3, mixtral
CTX_SIZE=16 # context size
DATA_TYPE=rel #semantic composition of the context
DOC_SET=rra # structural composition of the context
PROPMT_TYPE=default

CUDA_VISIBLE_DEVICES=6 python3 main.py \
  --input_path "./datasets/${CTX_SIZE}k_${DATA_TYPE}.json" \
  --output_path "./results/${MODEL}/${PROPMT_TYPE}/${CTX_SIZE}k_${DATA_TYPE}-${DOC_SET}_res.csv" \
  --model_type ${MODEL}  \
  --doc_setting ${DOC_SET} \
  --api_key "your-key"

```
#### Evaluation
```bash
python3 eval/evaluate.py \
  --input_file "./results/${MODEL}/${PROPMT_TYPE}/${CTX_SIZE}k_${DATA_TYPE}-${DOC_SET}_res.csv" \
  --output_file "./results/eval/${MODEL}/${PROPMT_TYPE}/${CTX_SIZE}k_${DATA_TYPE}-${DOC_SET}_evaluated.csv"
```


### 📖 Citation

```
@inproceedings{koo2024where,
title={Where am I? Large Language Models Wandering between Semantics and Structures in Long Contexts},
author={Seonmin Koo, Jinsung Kim, YoungJoon Jang, Chanjun Park, Heuiseok Lim},
booktitle={The 2024 Conference on Empirical Methods in Natural Language Processing},
year={2024},
url={https://openreview.net/forum?id=gY1eSVMx2E}
}
```

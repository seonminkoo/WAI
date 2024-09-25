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

First, by following the steps below, you can generate a dataset that encompasses various semantic and structural compositions of the context.

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
python eval/evaluation.py 
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

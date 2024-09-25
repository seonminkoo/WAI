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
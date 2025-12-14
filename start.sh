#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com  # 中国大陆加速（可选）
streamlit run app.py
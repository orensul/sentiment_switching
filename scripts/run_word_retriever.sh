#!/usr/bin/env bash

PROJECT_DIR_PATH="$PWD/$(dirname $0)/../"
cd ${PROJECT_DIR_PATH}

PYTHONPATH=${PROJECT_DIR_PATH} \
python -u sentiment_switching_model/word_retriever.py "$@"
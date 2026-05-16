# VN30 Stock Prediction Pipeline — Makefile
# Usage:
#   make run-pipeline BASE_PATH=./
#   make chat BASE_PATH=./
#   make train-tft BASE_PATH=./

BASE_PATH ?= ./

run-pipeline:
	python main.py --mode pipeline --base-path $(BASE_PATH)

chat:
	python main.py --mode chat --base-path $(BASE_PATH)

train-tft:
	python main.py --mode train-tft --base-path $(BASE_PATH)

.PHONY: install test lint format run pipeline validate help

# Default target
help:
	@echo "Available commands:"
	@echo "  make install  - Install requirements in active python environment"
	@echo "  make test     - Run test suite with pytest"
	@echo "  make lint     - Run ruff code linter check"
	@echo "  make format   - Run black formatter and ruff checks"
	@echo "  make run      - Launch Streamlit dashboard"
	@echo "  make pipeline - Run ingestion, modeling, and backtesting pipelines"
	@echo "  make validate - Run out-of-sample and walk-forward validation"

install:
	pip install -r requirements.txt

test:
	pytest

lint:
	ruff check src tests app

format:
	black src tests app
	ruff check --fix src tests app

run:
	streamlit run app/streamlit_app.py

pipeline:
	python run_pipeline.py
	python run_modeling.py
	python run_backtesting.py
	python run_validation.py

validate:
	python run_validation.py

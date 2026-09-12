# Breast Cancer CNN Classifier - Makefile
# Common development tasks

.PHONY: help install install-dev test lint format type-check clean \
        explore preprocess train evaluate dashboard report full \
        docker-build docker-run docker-train docker-eval docker-full \
        check-deps security-scan

# Default target
help:
	@echo "Breast Cancer CNN Classifier - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  install       Install production dependencies"
	@echo "  install-dev   Install development dependencies"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint          Run all linting (black, flake8, isort, mypy)"
	@echo "  format        Format code with black and isort"
	@echo "  type-check    Run mypy type checking"
	@echo "  test          Run unit tests with coverage"
	@echo "  check-deps    Audit dependencies for vulnerabilities"
	@echo "  security-scan Run bandit security scan"
	@echo ""
	@echo "Pipeline:"
	@echo "  explore       Run data exploration"
	@echo "  preprocess    Run data preprocessing"
	@echo "  train         Train model (default 100 epochs)"
	@echo "  evaluate      Evaluate trained model"
	@echo "  dashboard     Launch Streamlit dashboard"
	@echo "  report        Generate APA 7 technical report"
	@echo "  full          Run complete pipeline"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-run    Run dashboard in Docker"
	@echo "  docker-train  Run training in Docker"
	@echo "  docker-eval   Run evaluation in Docker"
	@echo "  docker-full   Run full pipeline in Docker"
	@echo ""
	@echo "Utilities:"
	@echo "  clean         Clean generated files"
	@echo "  reset         Reset project (clean + reinstall)"

# =====================================================================
# Setup
# =====================================================================

install:
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev: install
	pip install black flake8 isort mypy pytest pytest-cov pytest-mock pip-audit bandit

# =====================================================================
# Code Quality
# =====================================================================

lint: format-check flake8-check mypy-check

format:
	black src/ main.py tests/
	isort src/ main.py tests/

format-check:
	black --check src/ main.py tests/
	isort --check-only src/ main.py tests/

flake8-check:
	flake8 src/ main.py tests/

mypy-check:
	mypy src/ --ignore-missing-imports

type-check: mypy-check

test:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

check-deps:
	pip-audit -r requirements.txt

security-scan:
	bandit -r src/ -f json -o bandit-report.json || true
	@echo "Security report saved to bandit-report.json"

# =====================================================================
# Pipeline
# =====================================================================

explore:
	python main.py --mode explore

preprocess:
	python main.py --mode preprocess

train:
	python main.py --mode train --epochs $(or $(EPOCHS),100)

evaluate:
	python main.py --mode evaluate

dashboard:
	streamlit run src/dashboard/app.py

report:
	python main.py --mode report

full:
	python main.py --mode full

# =====================================================================
# Docker
# =====================================================================

docker-build:
	docker build -t breast-cancer-cnn:latest .

docker-run:
	docker run -d \
		-p 8501:8501 \
		-p 6006:6006 \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/assets:/app/assets \
		-v $(PWD)/logs:/app/logs \
		-v $(PWD)/reports:/app/reports \
		-v $(PWD)/mlruns:/app/mlruns \
		--name breast-cancer-dashboard \
		breast-cancer-cnn:latest

docker-train:
	docker run --rm --gpus all \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/assets:/app/assets \
		-v $(PWD)/logs:/app/logs \
		-v $(PWD)/reports:/app/reports \
		-v $(PWD)/mlruns:/app/mlruns \
		-v $(PWD)/configs:/app/configs \
		breast-cancer-cnn:latest \
		python main.py --mode train --epochs $(or $(EPOCHS),100)

docker-eval:
	docker run --rm --gpus all \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/assets:/app/assets \
		-v $(PWD)/logs:/app/logs \
		-v $(PWD)/reports:/app/reports \
		-v $(PWD)/mlruns:/app/mlruns \
		-v $(PWD)/configs:/app/configs \
		breast-cancer-cnn:latest \
		python main.py --mode evaluate

docker-full:
	docker run --rm --gpus all \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/assets:/app/assets \
		-v $(PWD)/logs:/app/logs \
		-v $(PWD)/reports:/app/reports \
		-v $(PWD)/mlruns:/app/mlruns \
		-v $(PWD)/configs:/app/configs \
		breast-cancer-cnn:latest \
		python main.py --mode full

docker-logs:
	docker logs -f breast-cancer-dashboard

docker-stop:
	docker stop breast-cancer-dashboard && docker rm breast-cancer-dashboard

# =====================================================================
# Docker Compose
# =====================================================================

compose-up:
	docker-compose up -d

compose-down:
	docker-compose down

compose-train:
	docker-compose --profile training up breast-cancer-train

compose-eval:
	docker-compose --profile evaluation up breast-cancer-eval

compose-full:
	docker-compose --profile full up breast-cancer-full

compose-mlflow:
	docker-compose --profile mlflow up -d mlflow

compose-monitoring:
	docker-compose --profile monitoring up -d tensorboard

# =====================================================================
# Utilities
# =====================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage bandit-report.json audit.json 2>/dev/null || true
	rm -rf logs/tensorboard/* 2>/dev/null || true

clean-all: clean
	rm -rf assets/models/* 2>/dev/null || true
	rm -rf reports/exploration/* 2>/dev/null || true
	rm -rf reports/technical_report.pdf 2>/dev/null || true
	rm -rf data/processed/* 2>/dev/null || true
	rm -rf mlruns/* 2>/dev/null || true

reset: clean-all
	rm -rf venv/ 2>/dev/null || true
	$(MAKE) install-dev

# =====================================================================
# Development Helpers
# =====================================================================

notebook:
	jupyter lab notebooks/

shell:
	docker run -it --rm --gpus all \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/assets:/app/assets \
		-v $(PWD)/src:/app/src \
		breast-cancer-cnn:latest bash

# Quick training with fewer epochs for testing
quick-train:
	python main.py --mode train --epochs 5

# Generate synthetic test data
gen-test-data:
	python -c "
import numpy as np
from PIL import Image
import os
os.makedirs('data/raw/Non-IDC', exist_ok=True)
os.makedirs('data/raw/IDC', exist_ok=True)
for i in range(50):
    img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    Image.fromarray(img).save(f'data/raw/Non-IDC/synth_{i}.png')
    img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    Image.fromarray(img).save(f'data/raw/IDC/synth_{i}.png')
print('Generated 50 synthetic images per class')
"
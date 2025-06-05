.PHONY: install test lint format clean build publish dev docker-build docker-run

# Variables
POETRY := poetry
PYTHON := python
PROJECT := colpali_server

# Installation
install:
	$(POETRY) install
	@echo "✅ Installation terminée"

# Qualité et formatage du code + détection d'erreurs
format:
	$(POETRY) run black --line-length 79 src/
	$(POETRY) run isort src/
	$(POETRY) run black --line-length 79 src/
	$(POETRY) run isort src/
	ruff format src/
	ruff check src/ --ignore D107 --fix --unsafe-fixes
	$(POETRY) run mypy src/

# Nettoyage
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
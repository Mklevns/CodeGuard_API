# CodeGuard API - Development Makefile

.PHONY: help install test lint format clean dev build deploy

help:
	@echo "CodeGuard API Development Commands"
	@echo "=================================="
	@echo "install      Install dependencies"
	@echo "test         Run test suite"
	@echo "lint         Run linting checks"
	@echo "format       Format code"
	@echo "clean        Clean temporary files"
	@echo "dev          Start development server"
	@echo "build        Build for production"
	@echo "deploy       Deploy to production"

install:
	pip install -e .
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v --cov=. --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

lint:
	flake8 . --max-line-length=88 --extend-ignore=E203,W503
	pylint *.py --disable=C0114,C0115,C0116
	mypy . --ignore-missing-imports

format:
	black . --line-length=88
	isort . --profile=black

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/

dev:
	python main.py

build:
	python setup.py sdist bdist_wheel

docker-build:
	docker build -t codeguard-api .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

deploy:
	@echo "Deploying to production..."
	@echo "Please ensure environment variables are set:"
	@echo "- CODEGUARD_API_KEY"
	@echo "- OPENAI_API_KEY"
	@echo "- DATABASE_URL"

# VS Code extension commands
ext-install:
	cd vscode-extension && npm install

ext-build:
	cd vscode-extension && npm run compile

ext-package:
	cd vscode-extension && npm run package

ext-test:
	cd vscode-extension && npm test

# Documentation
docs-serve:
	cd docs && python -m http.server 8000

# Security checks
security:
	bandit -r . -x tests/

# Performance benchmarks
benchmark:
	python -m pytest tests/performance/ -v --benchmark-only
.PHONY: help install dev-setup test lint format type-check clean docker-up docker-down

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies with Poetry"
	@echo "  make dev-setup     - Set up development environment"
	@echo "  make test          - Run tests with coverage"
	@echo "  make lint          - Run linting checks"
	@echo "  make format        - Format code with black and ruff"
	@echo "  make type-check    - Run mypy type checking"
	@echo "  make clean         - Clean up generated files"
	@echo "  make docker-up     - Start all services with docker-compose"
	@echo "  make docker-down   - Stop all services"

install:
	poetry install

dev-setup: install
	poetry run pre-commit install
	@echo "Development environment setup complete!"

test:
	poetry run pytest

lint:
	poetry run ruff check src tests
	poetry run black --check src tests

format:
	poetry run ruff check --fix src tests
	poetry run black src tests

type-check:
	poetry run mypy src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov .coverage dist build

docker-up:
	docker-compose up -d
	@echo "Waiting for services to be healthy..."
	@sleep 10
	docker-compose ps

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

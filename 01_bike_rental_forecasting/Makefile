.PHONY: install infra infra-down lakefs-repo mlflow dev test lint format help

# Load .env if it exists so all make targets have the environment variables.
-include .env
export

# macOS needs libomp on the dynamic linker path for XGBoost/LightGBM.
# Linux and Windows do not need this; DYLD stays empty and drops out.
UNAME := $(shell uname -s)
ifeq ($(UNAME), Darwin)
LIBOMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
ifneq ($(LIBOMP_PREFIX),)
DYLD := DYLD_LIBRARY_PATH="$(LIBOMP_PREFIX)/lib:$$DYLD_LIBRARY_PATH"
endif
endif

help:
	@echo ""
	@echo "  make install      install all dependencies (including dev)"
	@echo "  make infra        start LakeFS (Docker, runs in background)"
	@echo "  make infra-down   stop LakeFS"
	@echo "  make lakefs-repo  create the LakeFS repository (run once after infra)"
	@echo "  make mlflow       start the MLflow tracking server (separate terminal)"
	@echo "  make dev          start the Dagster development server"
	@echo "  make test         run the test suite"
	@echo "  make lint         check code style and formatting"
	@echo "  make format       auto-fix formatting and linting"
	@echo ""

install:
	uv sync --dev

# Starts LakeFS in the background, backed by local disk (see docker-compose.yml).
infra:
	docker compose up -d

infra-down:
	docker compose down

# Creates the LakeFS repository. Run once after `make infra`.
lakefs-repo:
	uv run python scripts/bootstrap_lakefs.py

# Starts the MLflow server backed by SQLite (registry needs a DB-backed store).
# Run in a separate terminal before starting Dagster.
mlflow:
	uv run mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--artifacts-destination ./mlartifacts \
		--host 127.0.0.1 \
		--port 5000

# Starts the Dagster development server. Requires MLflow and LakeFS running.
dev:
	$(DYLD) MLFLOW_TRACKING_URI=http://127.0.0.1:5000 uv run dg dev

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .
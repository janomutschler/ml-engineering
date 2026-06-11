.PHONY: install infra infra-down lakefs-repo quickstart mlflow dev api smoke test lint format help

# Load .env if it exists so all targets share one source of truth for config.
-include .env
export

# Defaults (overridable via .env). ?= only sets if not already defined.
MLFLOW_HOST ?= 127.0.0.1
MLFLOW_PORT ?= 5000
MLFLOW_BACKEND_STORE_URI ?= sqlite:///mlflow.db
MLFLOW_ARTIFACTS_DESTINATION ?= ./mlartifacts
API_HOST ?= 127.0.0.1
API_PORT ?= 8001

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
	@echo "  make quickstart   start LakeFS + create repo, then print next steps"
	@echo "  make infra        start LakeFS (Docker, runs in background)"
	@echo "  make infra-down   stop LakeFS"
	@echo "  make lakefs-repo  create the LakeFS repository (run once after infra)"
	@echo "  make mlflow       start the MLflow tracking server (separate terminal)"
	@echo "  make dev          start the Dagster development server"
	@echo "  make api          start the prediction API (FastAPI)"
	@echo "  make smoke        send an example request to a running API"
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

# One-shot setup: start LakeFS, wait for it, create the repository, then print
# the remaining steps. The long-running servers (MLflow, Dagster, API) each need
# their own terminal, so they are not started here.
quickstart: infra
	@echo "Waiting for LakeFS to become ready..."
	@sleep 5
	@$(MAKE) lakefs-repo
	@echo ""
	@echo "LakeFS is up and the repository is ready. Now, in separate terminals:"
	@echo "  1) make mlflow   (tracking server)"
	@echo "  2) make dev      (Dagster - materialize the pipeline to train + promote a model)"
	@echo "  3) make api      (prediction API, once a champion is promoted)"
	@echo ""

# Creates the LakeFS repository. Run once after `make infra`.
lakefs-repo:
	uv run python scripts/bootstrap_lakefs.py

# Starts the MLflow server backed by SQLite (registry needs a DB-backed store).
# Run in a separate terminal before starting Dagster.
mlflow:
	uv run mlflow server \
		--backend-store-uri $(MLFLOW_BACKEND_STORE_URI) \
		--artifacts-destination $(MLFLOW_ARTIFACTS_DESTINATION) \
		--host $(MLFLOW_HOST) \
		--port $(MLFLOW_PORT)

# Starts the Dagster development server. Requires MLflow and LakeFS running.
dev:
	$(DYLD) uv run dg dev

# Starts the prediction API. Requires MLflow (for the registry) and LakeFS.
api:
	$(DYLD) uv run uvicorn bike_rental.api.main:app --host $(API_HOST) --port $(API_PORT) --reload

# Live smoke check: sends an example request to a running API and prints the
# forecast. Requires the full stack up and a promoted champion model.
predict:
	uv run python scripts/api_example_request.py

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .
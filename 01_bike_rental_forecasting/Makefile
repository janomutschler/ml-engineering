.PHONY: install dev mlflow test lint format help
 
help:
	@echo ""
	@echo "  make install   install all dependencies (including dev)"
	@echo "  make mlflow    start the MLflow tracking server (run in a separate terminal)"
	@echo "  make dev       start the Dagster development server"
	@echo "  make test      run the test suite"
	@echo "  make lint      check code style and formatting"
	@echo "  make format    auto-fix formatting and linting"
	@echo ""
 
install:
	uv sync --dev
 
# Starts the MLflow server backed by a local SQLite database.
# The model registry requires a DB-backed store; the plain file store does not support it.
# Run this in a separate terminal before starting Dagster.
mlflow:
	uv run mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--artifacts-destination ./mlartifacts \
		--host 127.0.0.1 \
		--port 5000
 
# Detect macOS and set libomp path for XGBoost (Homebrew dependency).
# Linux and Windows do not need this.
UNAME := $(shell uname -s)
ifeq ($(UNAME), Darwin)
    LIBOMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
    ifneq ($(LIBOMP_PREFIX),)
        DYLD := DYLD_LIBRARY_PATH="$(LIBOMP_PREFIX)/lib:$$DYLD_LIBRARY_PATH"
    endif
endif

# Starts the Dagster development server.
# Requires the MLflow server to be running (make mlflow).
dev:
	$(DYLD) MLFLOW_TRACKING_URI=http://127.0.0.1:5000 uv run dg dev
 
test:
	$(DYLD) uv run pytest
 
lint:
	uv run ruff check .
	uv run ruff format --check .
 
format:
	uv run ruff format .
	uv run ruff check --fix .
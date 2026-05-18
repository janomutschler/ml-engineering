# ml-engineering

Building end-to-end machine learning systems and MLOps workflows - from data pipelines and model training to experiment tracking and deployment. Developed during the appliedAI ML & MLOps Track (ongoing).

## Structure

- `data/raw/`: source datasets used in the exercises.
- `notebooks/`: notebook workspaces for each week.
- `src/`: reusable Python code extracted from notebooks.
- `subjects/`: assignment descriptions and requirements.
- `reports/`: exported reports or written analysis.
- `tests/`: automated tests and fixtures.

## Current Focus

Week 1 covers a Titanic classification assignment with two parts: a scikit-learn logistic regression workflow and a from-scratch NumPy implementation.

More weeks will be added as their subject files become available.

## Quickstart

- This project uses `uv` together with `pyproject.toml` for environment and dependency management. `uv sync --dev` bootstraps an isolated environment automatically (no manual `venv` steps required).

```bash
# create/sync the project environment and install runtime + dev deps
uv sync --dev

# run tests via uv
uv run pytest

```


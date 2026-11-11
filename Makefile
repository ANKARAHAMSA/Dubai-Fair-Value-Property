.PHONY: setup data features train analyze api app lint test verify report clean

UV := uv
PY := $(UV) run python

setup:
	$(UV) sync --extra dev --python 3.12

data:
	$(PY) -m dfv.data.download
	$(PY) -m dfv.data.clean
	$(PY) -m dfv.data.validate

features:
	$(PY) -m dfv.features.build

train:
	$(PY) -m dfv.models.train

analyze:
	$(PY) -m dfv.analysis.run_all

api:
	$(UV) run uvicorn dfv.api.main:app --host 127.0.0.1 --port 8000 --reload

app:
	$(UV) run streamlit run src/dfv/app/Home.py

lint:
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests
	$(UV) run mypy src/dfv

test:
	$(UV) run pytest --cov=src/dfv --cov-report=term-missing

verify:
	$(PY) -m dfv.verify

report:
	$(PY) -m dfv.verify --report-only

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} +

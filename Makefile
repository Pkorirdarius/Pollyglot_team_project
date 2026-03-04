# ── Polyglot RAG — Developer Makefile (2026 Edition) ──────────────────────────
.PHONY: install run-api run-ui test lint clean ingest setup

# Colors for terminal output
BLUE := \033[34m
NC := \033[0m

install:
	@echo "$(BLUE)Installing dependencies...$(NC)"
	pip install -U pip
	pip install -r requirements.txt

setup:
	@echo "$(BLUE)Creating directory structure...$(NC)"
	mkdir -p data/vectorstore data/raw logs ui api models config utils tests
	@if [ ! -f .env ]; then echo "$(BLUE)Warning: .env file not found. Please create one based on the template.$(NC)"; fi

run-api:
	@echo "$(BLUE)Starting FastAPI Backend...$(NC)"
	export PYTHONPATH=$${PYTHONPATH}:. && uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	@echo "$(BLUE)Starting Streamlit UI...$(NC)"
	export PYTHONPATH=$${PYTHONPATH}:. && streamlit run ui/streamlit_app.py

test:
	@echo "$(BLUE)Running Smoke Tests...$(NC)"
	export PYTHONPATH=$${PYTHONPATH}:. && pytest tests/ -v

lint:
	@echo "$(BLUE)Linting with Ruff and Type Checking with Mypy...$(NC)"
	ruff check .
	mypy .

ingest:
	@echo "$(BLUE)Ingesting: $(FILE) as $(TYPE)$(NC)"
	@if [ -z "$(FILE)" ]; then echo "Error: FILE is not set. Use 'make ingest FILE=path/to/file TYPE=pdf'"; exit 1; fi
	export PYTHONPATH=$${PYTHONPATH}:. && python -c "from data_wrangling.loader import load_and_split; from data_wrangling.vectorstore import ingest_documents; ingest_documents(load_and_split('$(FILE)', '$(TYPE)'))"

clean:
	@echo "$(BLUE)Cleaning up temporary files...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache
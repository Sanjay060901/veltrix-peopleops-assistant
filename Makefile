PYTHON ?= python

.PHONY: install test run up

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

test:
	$(PYTHON) -m pytest -q

run:
	$(PYTHON) -m uvicorn app.main:app_factory --factory --host 0.0.0.0 --port 8000

up:
	docker compose up --build

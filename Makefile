PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
RUN := $(BIN)/python

.PHONY: setup check help run-following sync-following enrich-media session-from-browser session-status dashboard dashboard-bg dashboard-stop dashboard-status

setup:
	$(PYTHON) -m venv $(VENV)
	. $(BIN)/activate && pip install -r requirements.txt

check:
	$(PYTHON) -m compileall scripts/ig_pipeline.py
	. $(BIN)/activate && $(RUN) scripts/ig_pipeline.py --help
	. $(BIN)/activate && $(RUN) scripts/ig_pipeline.py run-following --help
	. $(BIN)/activate && $(RUN) scripts/ig_pipeline.py enrich-media --help

help:
	. $(BIN)/activate && $(RUN) scripts/ig_pipeline.py --help

run-following:
	. $(BIN)/activate && $(RUN) scripts/ig_pipeline.py run-following

sync-following:
	. $(BIN)/activate && $(RUN) scripts/ig_pipeline.py sync-following

enrich-media:
	. $(BIN)/activate && $(RUN) scripts/ig_pipeline.py enrich-media

session-from-browser:
	. $(BIN)/activate && $(RUN) scripts/ig_pipeline.py session-from-browser

session-status:
	. $(BIN)/activate && $(RUN) scripts/ig_pipeline.py session-status

dashboard:
	. $(BIN)/activate && streamlit run dashboard/app.py

dashboard-bg:
	mkdir -p .state
	nohup $(BIN)/streamlit run dashboard/app.py --server.headless true --server.port 8501 --server.address 127.0.0.1 > .state/dashboard.log 2>&1 & echo $$! > .state/dashboard.pid
	@echo "Dashboard started: http://127.0.0.1:8501 (pid $$(cat .state/dashboard.pid))"

dashboard-stop:
	@if test -f .state/dashboard.pid; then kill $$(cat .state/dashboard.pid) && rm -f .state/dashboard.pid && echo "Dashboard stopped."; else echo "No .state/dashboard.pid found."; fi

dashboard-status:
	@if test -f .state/dashboard.pid; then ps -p $$(cat .state/dashboard.pid) -o pid,command || true; else echo "No .state/dashboard.pid found."; fi

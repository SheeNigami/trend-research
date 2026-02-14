PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
RUN := $(BIN)/python

.PHONY: setup check help run-following sync-following enrich-media session-from-browser session-status

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

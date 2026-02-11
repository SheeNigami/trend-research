# Contributing

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Local Checks

Run these before opening a PR:

```bash
python3 -m compileall scripts/ig_pipeline.py
python scripts/ig_pipeline.py --help
python scripts/ig_pipeline.py run-following --help
python scripts/ig_pipeline.py enrich-media --help
```

## Pull Requests

- Keep PRs small and focused.
- Update `README.md` when behavior or commands change.
- Do not commit `.env`, `.secrets/`, `.state/`, or `data/`.
- If adding new dependencies, update `requirements.txt`.


# trend-research

Simple Instagram collection + enrichment pipeline.

It can:
- scrape all accounts followed by a seed profile
- keep only recent posts (you choose max age)
- download media (posts + reels by default)
- extract structured metadata
- transcribe videos (Whisper)
- OCR text from images and video frames

## 1. Setup (one time)

```bash
cd /Users/sheen/Desktop/work/trend-research
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or:

```bash
make setup
```

## 2. Login (one time)

```bash
python scripts/ig_pipeline.py login --username YOUR_IG_USERNAME
```

This creates `.secrets/instagram.session` (already gitignored).

## 3. Easiest way to run (recommended)

### Option A: pass values directly

```bash
python scripts/ig_pipeline.py run-following \
  --username YOUR_IG_USERNAME \
  --seed-profile PROFILE_OR_URL \
  --max-age-days 14
```

### Option B: set defaults in `.env` and run shorter command

Create `.env` from `.env.example`:

```bash
cp .env.example .env
```

Edit `.env`:

```env
IG_USERNAME=your_instagram_username
IG_SEED_PROFILE=natgeo
IG_MAX_AGE_DAYS=14
```

Then run:

```bash
python scripts/ig_pipeline.py run-following
```

Or:

```bash
make run-following
```

## 4. What gets collected

From Instaloader metadata (when present):
- post id + shortcode
- owner username + owner id
- timestamp (UTC)
- like/comment counts
- caption
- hashtags + mentions (parsed from caption)
- media type (image/video), video duration
- music attribution info (if Instagram provides it)

From enrichment:
- transcript for videos (`.transcript.json` + `.transcript.txt`)
- OCR text for images and video frames (`.ocr.txt`)
- audio stream probe for video (`ffprobe`)

## 5. Where data is stored

Raw downloads:
- `data/instagram/following/...`

Structured analysis output:
- `data/analysis/<relative_path>/<media_stem>.analysis.json`
- `data/analysis/<relative_path>/<media_stem>.transcript.json` (video)
- `data/analysis/<relative_path>/<media_stem>.transcript.txt` (video)
- `data/analysis/<relative_path>/<media_stem>.ocr.txt` (image/video)

Global catalog:
- `data/analysis/catalog.jsonl` (one JSON object per media file)

## 6. Most useful flags

`run-following`:
- `--max-age-days 7` only keep newer content
- `--reels-only` scrape reels only (default is posts + reels)
- `--no-transcribe` skip Whisper
- `--no-ocr` skip OCR
- `--overwrite` regenerate analysis files

Use dry-run to preview the scraping command:

```bash
python scripts/ig_pipeline.py run-following --seed-profile natgeo --dry-run
```

## 7. Optional dependencies for best enrichment

Install if you want full transcript/OCR support:
- `ffmpeg` and `ffprobe`
- `tesseract`
- `whisper` CLI (for example: `pip install openai-whisper`)

If these are missing, the pipeline still runs and marks that step as skipped in metadata.

## 8. Advanced commands (optional)

Scrape only:
```bash
python scripts/ig_pipeline.py sync-following --seed-profile natgeo --max-age-days 14
```

Enrich only:
```bash
python scripts/ig_pipeline.py enrich-media --input-root data/instagram/following --output-root data/analysis
```

## Notes

- Use this only for content you are authorized to collect and in line with Instagram terms.
- Keep `.secrets/instagram.session` private.

## Publish to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

This repo includes:
- CI workflow: `.github/workflows/ci.yml`
- issue templates and PR template
- `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`

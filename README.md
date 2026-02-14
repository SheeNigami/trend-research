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
  --max-age-days 30
```

### Option B: set defaults in `.env` and run shorter command

Create `.env` from `.env.example`:

```bash
cp .env.example .env
```

Edit `.env`:

```env
IG_USERNAME=your_instagram_username
IG_COOKIE_BROWSER=Chrome
IG_SEED_PROFILE=natgeo
IG_MAX_AGE_DAYS=30
IG_TARGET_MODE=following
IG_FEED_COUNT=80
IG_PUBLIC_CANDIDATES_FILE=config/public_following_candidates.txt
IG_INSTALOADER_RETRIES=2
IG_INSTALOADER_RETRY_WAIT_SECONDS=180
IG_INSTALOADER_RETRY_BACKOFF=1.5
```

Then run:

```bash
python scripts/ig_pipeline.py run-following
```

Or:

```bash
make run-following
```

## 3.1 Browser-based session refresh (recommended for login issues)

If login repeatedly returns `401` / `Please wait a few minutes...`, refresh session
from browser cookies:

```bash
pip install "instaloader[browser-cookie3]"
python scripts/ig_pipeline.py session-from-browser --username YOUR_IG_USERNAME --browser Chrome
python scripts/ig_pipeline.py session-status
```

If `session-status` reports `usable`, retry your login-required mode.

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
- `--target-mode feed` use your own feed instead of enumerating followees
- `--feed-count 80` max feed posts to inspect when using `--target-mode feed`
- `--target-mode public-candidates` no-login mode (seed profile + candidates file)
- `--stop-when-older` stop scanning each profile once `--max-age-days` is reached (default)
- `--no-stop-when-older` restore old behavior (`post-filter` + many `skipped` lines)
- `--public-candidates-file config/public_following_candidates.txt` source for no-login candidates
- `session-from-browser --browser Chrome` import working browser cookies into session file
- `session-status` verify local session contains non-empty `sessionid`
- `--instaloader-retries 3` retry temporary failures (rate limits / transient blocks)
- `--instaloader-retry-wait-seconds 240` base wait before retry
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
python scripts/ig_pipeline.py sync-following --seed-profile natgeo --max-age-days 30
```

Enrich only:
```bash
python scripts/ig_pipeline.py enrich-media --input-root data/instagram/following --output-root data/analysis
```

## Notes

- Use this only for content you are authorized to collect and in line with Instagram terms.
- Keep `.secrets/instagram.session` private.
- If Instagram returns `Please wait a few minutes before you try again.`, it is a temporary server-side throttle. The pipeline now retries automatically; you can tune retries via the flags/env vars above.
- If `@seed-profile` keeps failing with 401s, run with `--target-mode feed` as a practical fallback while cooldown clears.
- Instagram does not expose followee enumeration anonymously. `public-candidates` mode works without login by scraping the public seed profile plus usernames listed in `config/public_following_candidates.txt`, and it auto-skips private accounts.

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

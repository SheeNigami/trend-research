#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import lzma
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

SHORTCODE_RE = re.compile(
    r"(?:https?://)?(?:www\.)?instagram\.com/(?:reel|p|tv)/([A-Za-z0-9_-]+)"
)
BARE_SHORTCODE_RE = re.compile(r"^[A-Za-z0-9_-]{5,}$")
PROFILE_RE = re.compile(
    r"(?:https?://)?(?:www\.)?instagram\.com/([A-Za-z0-9._-]+)/?(?:\?.*)?$"
)
HASHTAG_RE = re.compile(r"#([A-Za-z0-9_]+)")
MENTION_RE = re.compile(r"@([A-Za-z0-9._]+)")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def ensure_instaloader_cli() -> None:
    if shutil.which("instaloader"):
        return
    raise SystemExit(
        "Instaloader CLI not found. Install deps with: pip install -r requirements.txt"
    )


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def env_float(name: str, fallback: float) -> float:
    value = os.getenv(name)
    if value is None:
        return fallback
    try:
        return float(value)
    except ValueError:
        return fallback


def read_non_comment_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def normalize_profiles(lines: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        username = line.lstrip("@").strip()
        if not username:
            continue
        if username in seen:
            continue
        seen.add(username)
        out.append(username)
    return out


def normalize_profile(value: str) -> str:
    cleaned = value.strip()
    match = PROFILE_RE.match(cleaned)
    if match:
        cleaned = match.group(1)
    return cleaned.lstrip("@").strip()


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def iter_media_files(root: Path) -> list[Path]:
    files: list[Path] = []
    if not root.exists():
        return files
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if is_video_file(path) or is_image_file(path):
            files.append(path)
    return sorted(files)


def maybe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        if path.suffix == ".xz":
            with lzma.open(path, "rt", encoding="utf-8") as handle:
                data = json.load(handle)
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def nested_get(obj: Any, path: list[str], default: Any = None) -> Any:
    current = obj
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def sidecar_base_stem(stem: str) -> str:
    match = re.match(r"^(.*)_\d+$", stem)
    if match:
        return match.group(1)
    return stem


def metadata_candidates_for_media(media_path: Path) -> list[Path]:
    base_stem = sidecar_base_stem(media_path.stem)
    candidates: list[Path] = []

    direct = [
        media_path.with_suffix(".json"),
        media_path.with_suffix(".json.xz"),
        media_path.with_name(f"{base_stem}.json"),
        media_path.with_name(f"{base_stem}.json.xz"),
    ]
    for item in direct:
        if item not in candidates:
            candidates.append(item)

    for nearby in sorted(media_path.parent.glob("*.json")):
        if nearby not in candidates:
            candidates.append(nearby)
    for nearby in sorted(media_path.parent.glob("*.json.xz")):
        if nearby not in candidates:
            candidates.append(nearby)
    return candidates


def load_post_metadata(media_path: Path) -> tuple[dict[str, Any] | None, Path | None]:
    for candidate in metadata_candidates_for_media(media_path):
        loaded = maybe_read_json(candidate)
        if loaded is not None:
            return loaded, candidate
    return None, None


def extract_post_node(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    node_candidates = [
        nested_get(metadata, ["node"]),
        nested_get(metadata, ["graphql", "shortcode_media"]),
        metadata,
    ]
    for candidate in node_candidates:
        if isinstance(candidate, dict):
            if any(
                key in candidate
                for key in (
                    "shortcode",
                    "is_video",
                    "date_utc",
                    "taken_at_timestamp",
                    "edge_media_to_caption",
                )
            ):
                return candidate
    return metadata


def extract_caption(node: dict[str, Any]) -> str:
    caption_edges = nested_get(node, ["edge_media_to_caption", "edges"], default=[])
    if isinstance(caption_edges, list) and caption_edges:
        first = caption_edges[0]
        if isinstance(first, dict):
            text = nested_get(first, ["node", "text"])
            if isinstance(text, str):
                return text
    raw_caption = node.get("caption")
    if isinstance(raw_caption, str):
        return raw_caption
    accessibility_caption = node.get("accessibility_caption")
    if isinstance(accessibility_caption, str):
        return accessibility_caption
    return ""


def parse_post_datetime(node: dict[str, Any], media_path: Path) -> str | None:
    value = node.get("date_utc")
    if isinstance(value, str):
        return value
    timestamp = node.get("taken_at_timestamp")
    if isinstance(timestamp, (int, float)):
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        return dt.isoformat()
    stem_match = re.match(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_UTC", media_path.stem)
    if not stem_match:
        return None
    try:
        dt = datetime.strptime(stem_match.group(1), "%Y-%m-%d_%H-%M-%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None
    return dt.isoformat()


def extract_music_info(node: dict[str, Any], metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    candidates = [
        node.get("clips_music_attribution_info"),
        nested_get(node, ["music_metadata"]),
        nested_get(node, ["audio"]),
        nested_get(metadata or {}, ["clips_music_attribution_info"]),
        nested_get(metadata or {}, ["node", "clips_music_attribution_info"]),
        nested_get(metadata or {}, ["graphql", "shortcode_media", "clips_music_attribution_info"]),
    ]
    for music in candidates:
        if not isinstance(music, dict):
            continue
        title = music.get("song_name") or music.get("title")
        artist = music.get("artist_name") or music.get("display_artist")
        return {
            "title": title if isinstance(title, str) else None,
            "artist": artist if isinstance(artist, str) else None,
            "is_original_audio": music.get("is_original_audio"),
            "raw": music,
        }
    return None


def safe_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def probe_audio_stream(video_path: Path) -> dict[str, Any]:
    if shutil.which("ffprobe") is None:
        return {"status": "skipped", "reason": "ffprobe not installed"}
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    result = run_command(cmd)
    if result.returncode != 0:
        return {
            "status": "error",
            "reason": result.stderr.strip() or "ffprobe failed",
        }
    has_audio = bool(result.stdout.strip())
    return {"status": "ok", "has_audio_stream": has_audio}


def transcribe_video_with_whisper(
    video_path: Path,
    transcript_json_path: Path,
    transcript_txt_path: Path,
    *,
    model: str,
    language: str | None,
    overwrite: bool,
) -> dict[str, Any]:
    if transcript_json_path.exists() and transcript_txt_path.exists() and not overwrite:
        try:
            loaded = json.loads(transcript_json_path.read_text(encoding="utf-8"))
            return {
                "status": "cached",
                "text": loaded.get("text"),
                "segments": loaded.get("segments"),
                "json_path": str(transcript_json_path),
                "text_path": str(transcript_txt_path),
            }
        except Exception:
            pass

    if shutil.which("whisper") is None:
        return {"status": "skipped", "reason": "whisper CLI not installed"}

    with tempfile.TemporaryDirectory(prefix="ig-whisper-") as tmpdir:
        cmd = [
            "whisper",
            str(video_path),
            "--model",
            model,
            "--output_dir",
            tmpdir,
            "--output_format",
            "json",
            "--fp16",
            "False",
        ]
        if language:
            cmd.extend(["--language", language])
        result = run_command(cmd)
        if result.returncode != 0:
            return {
                "status": "error",
                "reason": result.stderr.strip() or "whisper failed",
            }

        tmp_json = Path(tmpdir) / f"{video_path.stem}.json"
        if not tmp_json.exists():
            return {"status": "error", "reason": "whisper output JSON missing"}

        try:
            payload = json.loads(tmp_json.read_text(encoding="utf-8"))
        except Exception as exc:
            return {"status": "error", "reason": f"failed to read whisper output: {exc}"}

    transcript_json_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    transcript_text = payload.get("text", "") if isinstance(payload, dict) else ""
    safe_write_text(transcript_txt_path, transcript_text if isinstance(transcript_text, str) else "")
    return {
        "status": "ok",
        "text": transcript_text if isinstance(transcript_text, str) else "",
        "segments": payload.get("segments") if isinstance(payload, dict) else None,
        "json_path": str(transcript_json_path),
        "text_path": str(transcript_txt_path),
    }


def ocr_image_with_tesseract(image_path: Path, *, language: str) -> dict[str, Any]:
    if shutil.which("tesseract") is None:
        return {"status": "skipped", "reason": "tesseract not installed"}
    cmd = ["tesseract", str(image_path), "stdout", "-l", language]
    result = run_command(cmd)
    if result.returncode != 0:
        return {"status": "error", "reason": result.stderr.strip() or "tesseract failed"}
    text = result.stdout.strip()
    return {"status": "ok", "text": text}


def ocr_video_frames_with_tesseract(
    video_path: Path,
    *,
    language: str,
    frame_interval_seconds: float,
    max_frames: int,
) -> dict[str, Any]:
    if shutil.which("ffmpeg") is None:
        return {"status": "skipped", "reason": "ffmpeg not installed"}
    if shutil.which("tesseract") is None:
        return {"status": "skipped", "reason": "tesseract not installed"}

    with tempfile.TemporaryDirectory(prefix="ig-ocr-") as tmpdir:
        frames_pattern = str(Path(tmpdir) / "frame_%04d.jpg")
        ffmpeg_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"fps=1/{frame_interval_seconds}",
            "-frames:v",
            str(max_frames),
            frames_pattern,
            "-y",
        ]
        ffmpeg_result = run_command(ffmpeg_cmd)
        if ffmpeg_result.returncode != 0:
            return {"status": "error", "reason": ffmpeg_result.stderr.strip() or "ffmpeg failed"}

        frame_paths = sorted(Path(tmpdir).glob("frame_*.jpg"))
        if not frame_paths:
            return {"status": "ok", "text": "", "frames_sampled": 0}

        snippets: list[str] = []
        for frame in frame_paths:
            ocr_result = ocr_image_with_tesseract(frame, language=language)
            if ocr_result.get("status") != "ok":
                continue
            text = ocr_result.get("text")
            if isinstance(text, str) and text.strip():
                snippets.append(text.strip())

    combined = "\n\n".join(snippets).strip()
    return {"status": "ok", "text": combined, "frames_sampled": len(snippets)}


def write_analysis_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_shortcode(value: str) -> str | None:
    value = value.strip()
    if not value:
        return None
    match = SHORTCODE_RE.search(value)
    if match:
        return match.group(1)
    if BARE_SHORTCODE_RE.match(value):
        return value
    return None


def shortcodes_from_lines(lines: Iterable[str]) -> tuple[list[str], list[str]]:
    shortcodes: list[str] = []
    invalid: list[str] = []
    seen: set[str] = set()
    for item in lines:
        code = parse_shortcode(item)
        if not code:
            invalid.append(item)
            continue
        if code in seen:
            continue
        seen.add(code)
        shortcodes.append(code)
    return shortcodes, invalid


def print_cmd(cmd: list[str]) -> None:
    pretty = " ".join(shlex.quote(p) for p in cmd)
    print(f"$ {pretty}")


def run_instaloader(cmd: list[str], dry_run: bool = False) -> None:
    print_cmd(cmd)
    if dry_run:
        return
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def add_auth_flags(cmd: list[str], username: str | None, session_file: Path | None) -> None:
    if username:
        cmd.extend(["--login", username])
    if session_file:
        session_file.parent.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--sessionfile", str(session_file)])


def build_common_download_cmd(
    *,
    username: str | None,
    session_file: Path | None,
    dirname_pattern: str,
    quiet: bool,
) -> list[str]:
    cmd = [
        "instaloader",
        "--dirname-pattern",
        dirname_pattern,
        "--filename-pattern",
        "{date_utc}_UTC",
        "--sanitize-paths",
    ]
    add_auth_flags(cmd, username, session_file)
    if quiet:
        cmd.append("--quiet")
    return cmd


def command_login(args: argparse.Namespace) -> None:
    args.session_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["instaloader", "--login", args.username, "--sessionfile", str(args.session_file)]
    if args.quiet:
        cmd.append("--quiet")
    run_instaloader(cmd, dry_run=args.dry_run)
    print(f"Session file configured at {args.session_file}")


def command_sync_profiles(args: argparse.Namespace) -> None:
    if not args.username:
        print("No --username provided. Proceeding without login (public content only).")

    raw_profiles = read_non_comment_lines(args.profiles_file)
    profiles = normalize_profiles(raw_profiles)
    if not profiles:
        print(f"No profiles found in {args.profiles_file}")
        return

    args.download_root.mkdir(parents=True, exist_ok=True)
    args.latest_stamps.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_common_download_cmd(
        username=args.username,
        session_file=args.session_file,
        dirname_pattern=str(args.download_root / "{target}"),
        quiet=args.quiet,
    )
    cmd.extend(["--latest-stamps", str(args.latest_stamps)])
    if args.fast_update:
        cmd.append("--fast-update")
    if args.include_posts:
        cmd.append("--reels")
    else:
        cmd.extend(["--reels", "--no-posts"])
    cmd.extend(profiles)
    run_instaloader(cmd, dry_run=args.dry_run)


def command_sync_following(args: argparse.Namespace) -> None:
    if not args.username:
        raise SystemExit("--username is required for sync-following.")

    seed_profile = normalize_profile(args.seed_profile)
    if not seed_profile:
        raise SystemExit("--seed-profile must be a valid Instagram username.")
    if args.max_age_days < 0:
        raise SystemExit("--max-age-days must be >= 0.")

    args.download_root.mkdir(parents=True, exist_ok=True)
    args.latest_stamps.parent.mkdir(parents=True, exist_ok=True)

    max_age_seconds = args.max_age_days * 24 * 60 * 60
    min_unix_ts = int((datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)).timestamp())

    cmd = build_common_download_cmd(
        username=args.username,
        session_file=args.session_file,
        dirname_pattern=str(args.download_root / "{target}"),
        quiet=args.quiet,
    )
    cmd.extend(
        [
            "--latest-stamps",
            str(args.latest_stamps),
            "--post-filter",
            f"date_utc.timestamp() >= {min_unix_ts}",
            "--no-profile-pic",
        ]
    )
    if args.fast_update:
        cmd.append("--fast-update")
    if args.reels_only:
        cmd.extend(["--reels", "--no-posts"])
    else:
        cmd.append("--reels")
    cmd.append(f"@{seed_profile}")
    run_instaloader(cmd, dry_run=args.dry_run)


def command_download_links(args: argparse.Namespace) -> None:
    if not args.username:
        print("No --username provided. Proceeding without login (public content only).")

    lines = read_non_comment_lines(args.links_file)
    shortcodes, invalid = shortcodes_from_lines(lines)
    for bad in invalid:
        print(f"Skipping invalid link/shortcode: {bad}")
    if not shortcodes:
        print(f"No valid links found in {args.links_file}")
        return

    links_root = args.download_root / "links"
    links_root.mkdir(parents=True, exist_ok=True)

    cmd = build_common_download_cmd(
        username=args.username,
        session_file=args.session_file,
        dirname_pattern=str(links_root / "{target}"),
        quiet=args.quiet,
    )
    cmd.append("--")
    cmd.extend([f"-{code}" for code in shortcodes])
    run_instaloader(cmd, dry_run=args.dry_run)


def command_enrich_media(args: argparse.Namespace) -> None:
    media_root = args.input_root
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    media_files = iter_media_files(media_root)
    if args.limit is not None:
        media_files = media_files[: args.limit]

    if not media_files:
        print(f"No media files found under {media_root}")
        return

    processed = 0
    skipped = 0
    records: list[dict[str, Any]] = []
    for media_path in media_files:
        relative_media = media_path.relative_to(media_root)
        output_dir = output_root / relative_media.parent
        base_name = media_path.stem
        analysis_path = output_dir / f"{base_name}.analysis.json"
        transcript_json_path = output_dir / f"{base_name}.transcript.json"
        transcript_txt_path = output_dir / f"{base_name}.transcript.txt"
        ocr_txt_path = output_dir / f"{base_name}.ocr.txt"

        if analysis_path.exists() and not args.overwrite:
            skipped += 1
            continue

        metadata, metadata_path = load_post_metadata(media_path)
        node = extract_post_node(metadata)
        caption_text = extract_caption(node)
        hashtags = sorted(set(HASHTAG_RE.findall(caption_text)))
        mentions = sorted(set(MENTION_RE.findall(caption_text)))
        media_type = "video" if is_video_file(media_path) else "image"
        audio_probe = probe_audio_stream(media_path) if media_type == "video" else None

        record: dict[str, Any] = {
            "source_media_path": str(media_path),
            "relative_media_path": str(relative_media),
            "media_type": media_type,
            "downloaded_file": media_path.name,
            "source_metadata_file": str(metadata_path) if metadata_path else None,
            "post": {
                "shortcode": node.get("shortcode"),
                "id": node.get("id"),
                "owner_username": nested_get(node, ["owner", "username"]),
                "owner_id": nested_get(node, ["owner", "id"]),
                "timestamp_utc": parse_post_datetime(node, media_path),
                "like_count": node.get("edge_media_preview_like", {}).get("count")
                if isinstance(node.get("edge_media_preview_like"), dict)
                else node.get("like_count"),
                "comment_count": node.get("edge_media_to_comment", {}).get("count")
                if isinstance(node.get("edge_media_to_comment"), dict)
                else node.get("comment_count"),
                "caption": caption_text,
                "hashtags": hashtags,
                "mentions": mentions,
                "is_video": bool(node.get("is_video", media_type == "video")),
                "video_duration": node.get("video_duration"),
            },
            "music": extract_music_info(node, metadata),
            "transcription": {"status": "disabled"},
            "ocr": {"status": "disabled"},
            "audio_probe": audio_probe,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        if args.transcribe and media_type == "video":
            transcription = transcribe_video_with_whisper(
                media_path,
                transcript_json_path,
                transcript_txt_path,
                model=args.whisper_model,
                language=args.whisper_language,
                overwrite=args.overwrite,
            )
            record["transcription"] = transcription
        elif media_type == "video":
            record["transcription"] = {"status": "skipped", "reason": "transcription disabled"}

        if args.ocr:
            if media_type == "image":
                ocr_result = ocr_image_with_tesseract(media_path, language=args.ocr_language)
            else:
                ocr_result = ocr_video_frames_with_tesseract(
                    media_path,
                    language=args.ocr_language,
                    frame_interval_seconds=args.video_ocr_interval_seconds,
                    max_frames=args.video_ocr_max_frames,
                )
            record["ocr"] = ocr_result
            text_value = ocr_result.get("text")
            if isinstance(text_value, str):
                safe_write_text(ocr_txt_path, text_value)
                record["ocr"]["text_path"] = str(ocr_txt_path)
        else:
            record["ocr"] = {"status": "skipped", "reason": "ocr disabled"}

        write_analysis_json(analysis_path, record)
        records.append(record)
        processed += 1

    args.catalog_file.parent.mkdir(parents=True, exist_ok=True)
    with args.catalog_file.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    print(
        f"Enrichment complete. processed={processed}, skipped={skipped}, catalog={args.catalog_file}"
    )


def command_run_following(args: argparse.Namespace) -> None:
    sync_args = argparse.Namespace(
        username=args.username,
        session_file=args.session_file,
        seed_profile=args.seed_profile,
        max_age_days=args.max_age_days,
        download_root=args.download_root,
        latest_stamps=args.latest_stamps,
        quiet=args.quiet,
        dry_run=args.dry_run,
        fast_update=args.fast_update,
        reels_only=args.reels_only,
    )
    command_sync_following(sync_args)
    if args.dry_run:
        print("Dry-run complete for sync-following. Enrichment skipped in dry-run mode.")
        return

    enrich_args = argparse.Namespace(
        input_root=args.download_root,
        output_root=args.output_root,
        catalog_file=args.catalog_file,
        transcribe=args.transcribe,
        whisper_model=args.whisper_model,
        whisper_language=args.whisper_language,
        ocr=args.ocr,
        ocr_language=args.ocr_language,
        video_ocr_interval_seconds=args.video_ocr_interval_seconds,
        video_ocr_max_frames=args.video_ocr_max_frames,
        overwrite=args.overwrite,
        limit=args.limit,
    )
    command_enrich_media(enrich_args)


def command_run(args: argparse.Namespace) -> None:
    sync_args = argparse.Namespace(
        username=args.username,
        session_file=args.session_file,
        profiles_file=args.profiles_file,
        latest_stamps=args.latest_stamps,
        download_root=args.download_root,
        quiet=args.quiet,
        dry_run=args.dry_run,
        fast_update=args.fast_update,
        include_posts=args.include_posts,
    )
    links_args = argparse.Namespace(
        username=args.username,
        session_file=args.session_file,
        links_file=args.links_file,
        download_root=args.download_root,
        quiet=args.quiet,
        dry_run=args.dry_run,
    )
    command_sync_profiles(sync_args)
    command_download_links(links_args)


def base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Automate Instagram reel collection with Instaloader.\n"
            "Workflow: login once -> sync highlighted profiles -> optional ad-hoc reel links."
        )
    )
    return parser


def add_auth_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--username",
        default=os.getenv("IG_USERNAME"),
        help="Instagram username. Falls back to IG_USERNAME env var.",
    )
    parser.add_argument(
        "--session-file",
        type=Path,
        default=Path(".secrets/instagram.session"),
        help="Path to Instaloader session file.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce Instaloader output.")


def add_dry_run_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running.")


def add_following_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--seed-profile",
        default=os.getenv("IG_SEED_PROFILE"),
        help=(
            "Profile to inspect (scrapes all accounts this profile follows). "
            "Falls back to IG_SEED_PROFILE env var."
        ),
    )
    parser.add_argument(
        "--max-age-days",
        type=float,
        default=env_float("IG_MAX_AGE_DAYS", 30.0),
        help=(
            "Only download posts newer than this many days (default: 30). "
            "Falls back to IG_MAX_AGE_DAYS env var."
        ),
    )
    parser.add_argument(
        "--download-root",
        type=Path,
        default=Path("data/instagram/following"),
        help="Root directory for downloaded media.",
    )
    parser.add_argument(
        "--latest-stamps",
        type=Path,
        default=Path(".state/latest-stamps-following.ini"),
        help="State file used to only fetch new content.",
    )
    parser.add_argument(
        "--fast-update",
        action="store_true",
        help="Stop when first already-downloaded post appears for each followee.",
    )
    parser.add_argument(
        "--reels-only",
        action="store_true",
        help="Scrape reels only. By default, both regular posts and reels are scraped.",
    )


def add_profile_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profiles-file",
        type=Path,
        default=Path("config/profiles.txt"),
        help="Text file with one profile per line.",
    )
    parser.add_argument(
        "--download-root",
        type=Path,
        default=Path("data/instagram"),
        help="Root directory for downloaded media.",
    )
    parser.add_argument(
        "--latest-stamps",
        type=Path,
        default=Path(".state/latest-stamps.ini"),
        help="State file used to only fetch new content.",
    )
    parser.add_argument(
        "--fast-update",
        action="store_true",
        help="Stop when first already-downloaded post appears (faster, slightly less thorough).",
    )
    parser.add_argument(
        "--include-posts",
        action="store_true",
        help="Include regular posts in addition to reels.",
    )


def add_link_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--links-file",
        type=Path,
        default=Path("config/reel_urls.txt"),
        help="Text file with one Instagram reel/post URL or shortcode per line.",
    )
    parser.add_argument(
        "--download-root",
        type=Path,
        default=Path("data/instagram"),
        help="Root directory for downloaded media.",
    )


def create_cli() -> argparse.ArgumentParser:
    parser = base_parser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    login_parser = subparsers.add_parser(
        "login", help="Create/update local Instagram session file."
    )
    login_parser.add_argument("--username", required=True, help="Instagram username for login.")
    login_parser.add_argument(
        "--session-file",
        type=Path,
        default=Path(".secrets/instagram.session"),
        help="Path to write the session file.",
    )
    login_parser.add_argument("--quiet", action="store_true", help="Reduce Instaloader output.")
    add_dry_run_argument(login_parser)
    login_parser.set_defaults(handler=command_login)

    sync_parser = subparsers.add_parser(
        "sync-profiles", help="Download new content for profiles in config/profiles.txt."
    )
    add_auth_arguments(sync_parser)
    add_profile_arguments(sync_parser)
    add_dry_run_argument(sync_parser)
    sync_parser.set_defaults(handler=command_sync_profiles)

    following_parser = subparsers.add_parser(
        "sync-following",
        help="Download content from every account followed by the seed profile.",
    )
    add_auth_arguments(following_parser)
    add_following_arguments(following_parser)
    add_dry_run_argument(following_parser)
    following_parser.set_defaults(handler=command_sync_following)

    enrich_parser = subparsers.add_parser(
        "enrich-media",
        help=(
            "Process downloaded images/videos into structured metadata, transcripts, "
            "and OCR text artifacts."
        ),
    )
    enrich_parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/instagram/following"),
        help="Root folder containing downloaded Instagram media.",
    )
    enrich_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/analysis"),
        help="Where enrichment artifacts and sidecar JSON files are written.",
    )
    enrich_parser.add_argument(
        "--catalog-file",
        type=Path,
        default=Path("data/analysis/catalog.jsonl"),
        help="JSONL catalog file containing one enriched record per media file.",
    )
    enrich_parser.add_argument(
        "--transcribe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable Whisper transcription for video files.",
    )
    enrich_parser.add_argument(
        "--whisper-model",
        default="base",
        help="Whisper model name used for transcription.",
    )
    enrich_parser.add_argument(
        "--whisper-language",
        default=None,
        help="Optional Whisper language code (example: en).",
    )
    enrich_parser.add_argument(
        "--ocr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable OCR on images and sampled video frames.",
    )
    enrich_parser.add_argument(
        "--ocr-language",
        default="eng",
        help="Tesseract language code for OCR.",
    )
    enrich_parser.add_argument(
        "--video-ocr-interval-seconds",
        type=float,
        default=2.5,
        help="How often to sample frames from videos for OCR.",
    )
    enrich_parser.add_argument(
        "--video-ocr-max-frames",
        type=int,
        default=24,
        help="Maximum sampled frames per video for OCR.",
    )
    enrich_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute outputs even if analysis sidecars already exist.",
    )
    enrich_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of media files to process.",
    )
    enrich_parser.set_defaults(handler=command_enrich_media)

    run_following_parser = subparsers.add_parser(
        "run-following",
        help="One command: scrape followees from a seed profile, then enrich media.",
    )
    add_auth_arguments(run_following_parser)
    add_following_arguments(run_following_parser)
    run_following_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/analysis"),
        help="Where enrichment artifacts and sidecar JSON files are written.",
    )
    run_following_parser.add_argument(
        "--catalog-file",
        type=Path,
        default=Path("data/analysis/catalog.jsonl"),
        help="JSONL catalog file containing one enriched record per media file.",
    )
    run_following_parser.add_argument(
        "--transcribe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable Whisper transcription for video files.",
    )
    run_following_parser.add_argument(
        "--whisper-model",
        default="base",
        help="Whisper model name used for transcription.",
    )
    run_following_parser.add_argument(
        "--whisper-language",
        default=None,
        help="Optional Whisper language code (example: en).",
    )
    run_following_parser.add_argument(
        "--ocr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable OCR on images and sampled video frames.",
    )
    run_following_parser.add_argument(
        "--ocr-language",
        default="eng",
        help="Tesseract language code for OCR.",
    )
    run_following_parser.add_argument(
        "--video-ocr-interval-seconds",
        type=float,
        default=2.5,
        help="How often to sample frames from videos for OCR.",
    )
    run_following_parser.add_argument(
        "--video-ocr-max-frames",
        type=int,
        default=24,
        help="Maximum sampled frames per video for OCR.",
    )
    run_following_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute outputs even if analysis sidecars already exist.",
    )
    run_following_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of media files to process.",
    )
    add_dry_run_argument(run_following_parser)
    run_following_parser.set_defaults(handler=command_run_following)

    links_parser = subparsers.add_parser(
        "download-links", help="Download reels/posts from config/reel_urls.txt."
    )
    add_auth_arguments(links_parser)
    add_link_arguments(links_parser)
    add_dry_run_argument(links_parser)
    links_parser.set_defaults(handler=command_download_links)

    run_parser = subparsers.add_parser(
        "run", help="Run profile sync and then ad-hoc reel link download."
    )
    add_auth_arguments(run_parser)
    add_profile_arguments(run_parser)
    run_parser.add_argument(
        "--links-file",
        type=Path,
        default=Path("config/reel_urls.txt"),
        help="Text file with one Instagram reel/post URL or shortcode per line.",
    )
    add_dry_run_argument(run_parser)
    run_parser.set_defaults(handler=command_run)

    return parser


def main() -> int:
    load_dotenv()
    parser = create_cli()
    args = parser.parse_args()
    if args.command in {
        "login",
        "sync-profiles",
        "sync-following",
        "download-links",
        "run",
        "run-following",
    }:
        ensure_instaloader_cli()
    args.handler(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)

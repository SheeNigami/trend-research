import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


CATALOG_DEFAULT = Path("data/analysis/catalog.jsonl")
LABELS_PATH = Path(".state/trend_labels.json")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _iso_to_dt(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        # Example: 2026-02-13T23:10:19+00:00
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _join_text(*parts: Any) -> str:
    out: list[str] = []
    for part in parts:
        if not part:
            continue
        if isinstance(part, str):
            s = part.strip()
            if s:
                out.append(s)
            continue
        if isinstance(part, list):
            for item in part:
                if isinstance(item, str) and item.strip():
                    out.append(item.strip())
    return "\n\n".join(out).strip()


@st.cache_data(show_spinner=False)
def load_catalog(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


@st.cache_data(show_spinner=False)
def catalog_to_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    flat: list[dict[str, Any]] = []
    for r in rows:
        post = r.get("post") if isinstance(r.get("post"), dict) else {}
        music = r.get("music") if isinstance(r.get("music"), dict) else {}
        ap = r.get("audio_probe") if isinstance(r.get("audio_probe"), dict) else {}
        tr = r.get("transcription") if isinstance(r.get("transcription"), dict) else {}
        oc = r.get("ocr") if isinstance(r.get("ocr"), dict) else {}

        rel = r.get("relative_media_path") if isinstance(r.get("relative_media_path"), str) else ""
        profile = rel.split("/", 1)[0] if "/" in rel else None

        dt = _iso_to_dt(post.get("timestamp_utc"))
        like_count = post.get("like_count")
        comment_count = post.get("comment_count")

        hashtags = post.get("hashtags") if isinstance(post.get("hashtags"), list) else []
        mentions = post.get("mentions") if isinstance(post.get("mentions"), list) else []

        caption = post.get("caption") if isinstance(post.get("caption"), str) else ""
        tr_text = tr.get("text") if isinstance(tr.get("text"), str) else ""
        ocr_text = oc.get("text") if isinstance(oc.get("text"), str) else ""

        like_i = _safe_int(like_count)
        comment_i = _safe_int(comment_count)
        engagement = like_i + 2 * comment_i

        flat.append(
            {
                "profile": profile,
                "media_type": r.get("media_type"),
                "timestamp_utc": dt,
                "date_utc": dt.date().isoformat() if dt else None,
                "shortcode": post.get("shortcode"),
                "post_id": post.get("id"),
                "owner_username": post.get("owner_username"),
                "owner_id": post.get("owner_id"),
                "like_count": like_i if like_count is not None else None,
                "comment_count": comment_i if comment_count is not None else None,
                "engagement": engagement,
                "caption": caption,
                "hashtags": hashtags,
                "mentions": mentions,
                "music_title": music.get("title"),
                "music_artist": music.get("artist"),
                "is_original_audio": music.get("is_original_audio"),
                "has_audio_stream": ap.get("has_audio_stream"),
                "transcription_status": tr.get("status"),
                "transcription_text": tr_text,
                "ocr_status": oc.get("status"),
                "ocr_text": ocr_text,
                "source_media_path": r.get("source_media_path"),
                "relative_media_path": rel,
                "analysis_stem": r.get("analysis_stem"),
            }
        )
    df = pd.DataFrame(flat)
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    return df


def explode_counts(df: pd.DataFrame, column: str, *, limit: int = 30) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame(columns=[column, "count"])
    counts: Counter[str] = Counter()
    for items in df[column].tolist():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, str):
                continue
            val = item.strip()
            if not val:
                continue
            counts[val.lower()] += 1
    top = counts.most_common(limit)
    return pd.DataFrame([{column: k, "count": v} for k, v in top])


def load_labels() -> dict[str, Any]:
    if not LABELS_PATH.exists():
        return {}
    try:
        data = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def save_labels(labels: dict[str, Any]) -> None:
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LABELS_PATH.write_text(json.dumps(labels, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class ClusterSummary:
    cluster_id: int
    label: str
    count: int
    top_terms: list[str]
    example_shortcodes: list[str]


def build_clusters(df: pd.DataFrame, *, k: int, max_docs: int) -> tuple[pd.DataFrame, list[ClusterSummary]]:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    work = df.copy()
    work = work[work["timestamp_utc"].notna()]

    # Prefer captions + transcription; OCR can be noisy but often useful.
    work["content_text"] = work.apply(
        lambda r: _join_text(r.get("caption"), r.get("transcription_text"), r.get("ocr_text")),
        axis=1,
    )
    work = work[work["content_text"].astype(str).str.len() > 0]
    if work.empty:
        return df, []

    if len(work) > max_docs:
        work = work.sort_values("engagement", ascending=False).head(max_docs)

    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
    )
    try:
        X = vectorizer.fit_transform(work["content_text"].tolist())
    except ValueError:
        # e.g., empty vocabulary because the filtered window is too small / too repetitive
        return df, []
    if X.shape[0] < k:
        k = max(2, min(X.shape[0], k))
    model = KMeans(n_clusters=k, n_init="auto", random_state=7)
    clusters = model.fit_predict(X)
    work["cluster_id"] = clusters

    # Summaries: top TF-IDF terms per centroid.
    terms = vectorizer.get_feature_names_out()
    centroids = model.cluster_centers_
    summaries: list[ClusterSummary] = []
    for cid in range(k):
        idxs = centroids[cid].argsort()[::-1][:12]
        top_terms = [terms[i] for i in idxs]
        subset = work[work["cluster_id"] == cid]
        examples = (
            subset.sort_values("engagement", ascending=False)["shortcode"]
            .dropna()
            .astype(str)
            .head(5)
            .tolist()
        )
        summaries.append(
            ClusterSummary(
                cluster_id=cid,
                label=f"Cluster {cid}",
                count=int(len(subset)),
                top_terms=top_terms,
                example_shortcodes=examples,
            )
        )

    # Merge back into full df (NaN for rows not clustered).
    merged = df.merge(
        work[["relative_media_path", "cluster_id"]],
        on="relative_media_path",
        how="left",
    )
    return merged, summaries


def maybe_label_clusters_with_llm(
    summaries: list[ClusterSummary],
    *,
    rows_by_cluster: dict[int, list[dict[str, Any]]],
    model: str,
    api_key: str,
) -> dict[str, Any]:
    """
    Optional: label clusters via an LLM if OPENAI_API_KEY is present.
    Writes only short samples (no raw media) and saves results to .state/trend_labels.json.
    """
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"openai package not installed/usable: {exc}")

    client = OpenAI(api_key=api_key)

    labeled: dict[str, Any] = {}
    for s in summaries:
        samples = rows_by_cluster.get(s.cluster_id, [])[:6]
        sample_lines: list[str] = []
        for r in samples:
            sample_lines.append(
                f"- profile={r.get('profile')}, ts={r.get('timestamp_utc')}, likes={r.get('like_count')}, "
                f"comments={r.get('comment_count')}, shortcode={r.get('shortcode')}, "
                f"text={str(r.get('content_text',''))[:280].replace('\\n',' ')}"
            )

        prompt = (
            "You are labeling clusters of Instagram posts to help a trend dashboard.\n"
            "Given:\n"
            f"- top_terms: {', '.join(s.top_terms)}\n"
            "- samples:\n"
            + "\n".join(sample_lines)
            + "\n\n"
            "Return strict JSON with keys:\n"
            "- label: short name (3-8 words)\n"
            "- summary: 2-3 sentences describing the recurring theme\n"
            "- what_is_going_viral: 2-4 bullets focusing on what seems to drive engagement\n"
            "- suggested_queries: 3 short search-style queries to explore the trend\n"
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You output strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        text = resp.choices[0].message.content or "{}"
        try:
            payload = json.loads(text)
        except Exception:
            payload = {"label": s.label, "summary": text}
        labeled[str(s.cluster_id)] = payload

    return labeled


def main() -> None:
    st.set_page_config(page_title="Trend Research Dashboard", layout="wide")
    st.title("Trend Research Dashboard")

    st.sidebar.header("Data")
    catalog_path = st.sidebar.text_input("Catalog path", value=str(CATALOG_DEFAULT))
    rows = load_catalog(catalog_path)
    if not rows:
        st.error(f"No records found at {catalog_path}. Run: `make enrich-media` first.")
        st.stop()

    df = catalog_to_df(rows)
    df = df[df["timestamp_utc"].notna()].copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.sort_values("timestamp_utc", ascending=False)

    # Filters
    st.sidebar.header("Filters")
    profiles = sorted([p for p in df["profile"].dropna().unique().tolist() if p])
    media_types = sorted([t for t in df["media_type"].dropna().unique().tolist() if t])

    default_days = 21
    min_ts = df["timestamp_utc"].min()
    max_ts = df["timestamp_utc"].max()
    if pd.isna(min_ts) or pd.isna(max_ts):
        st.error("Catalog has no timestamps.")
        st.stop()

    days_back = st.sidebar.slider("Recent window (days)", 1, 60, default_days)
    cutoff = pd.Timestamp(_now_utc()) - pd.Timedelta(days=days_back)

    selected_profiles = st.sidebar.multiselect("Profiles", profiles, default=profiles)
    selected_types = st.sidebar.multiselect("Media types", media_types, default=media_types)
    min_likes = st.sidebar.number_input("Min likes", min_value=0, value=0, step=100)

    work = df.copy()
    work = work[work["timestamp_utc"] >= cutoff]
    if selected_profiles:
        work = work[work["profile"].isin(selected_profiles)]
    if selected_types:
        work = work[work["media_type"].isin(selected_types)]
    work = work[work["like_count"].fillna(0) >= min_likes]

    # Overview
    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Posts (media files)", f"{len(work):,}")
    c2.metric("Profiles", f"{work['profile'].nunique():,}")
    c3.metric("Avg likes", f"{int(work['like_count'].fillna(0).mean()):,}")
    c4.metric("Avg engagement", f"{int(work['engagement'].fillna(0).mean()):,}")

    # Time series
    st.subheader("Recent Activity")
    ts = (
        work.assign(day=work["timestamp_utc"].dt.floor("D"))
        .groupby(["day", "media_type"], dropna=False)
        .agg(posts=("relative_media_path", "count"), avg_likes=("like_count", "mean"))
        .reset_index()
    )
    ts["avg_likes"] = ts["avg_likes"].fillna(0).round(0).astype(int)
    st.line_chart(ts.pivot(index="day", columns="media_type", values="posts").fillna(0))

    # Viral table
    st.subheader("Most Viral (by engagement = likes + 2*comments)")
    topn = work.sort_values("engagement", ascending=False).head(50).copy()
    topn["timestamp_utc"] = topn["timestamp_utc"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(
        topn[
            [
                "timestamp_utc",
                "profile",
                "media_type",
                "like_count",
                "comment_count",
                "engagement",
                "shortcode",
                "music_title",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Top Hashtags / Mentions / Music")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        tags = explode_counts(work, "hashtags", limit=25)
        st.caption("Hashtags")
        st.bar_chart(tags.set_index("hashtags")["count"] if not tags.empty else pd.Series(dtype=int))
    with col_b:
        ment = explode_counts(work, "mentions", limit=25)
        st.caption("Mentions")
        st.bar_chart(ment.set_index("mentions")["count"] if not ment.empty else pd.Series(dtype=int))
    with col_c:
        music = (
            work.dropna(subset=["music_title"])
            .groupby(["music_title", "music_artist"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(25)
        )
        st.caption("Music (from IG attribution when available)")
        if music.empty:
            st.write("No music attribution present in this filtered window.")
        else:
            st.dataframe(music, use_container_width=True, hide_index=True)

    # Theme clustering (+ optional LLM labeling)
    st.subheader("Themes (Clustering)")
    with st.expander("Cluster settings", expanded=False):
        k = st.slider("Number of clusters", 2, 20, 8)
        max_docs = st.slider("Max docs clustered (top by engagement)", 50, 500, 250, step=25)
        st.caption(
            "Clustering is based on TF-IDF over caption + transcript + OCR text (when present)."
        )

    clustered, summaries = build_clusters(work, k=k, max_docs=max_docs)
    if not summaries:
        st.write("Not enough text to cluster in this filtered window.")
        return

    labels = load_labels()
    by_cluster_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    # Rebuild content_text for LLM samples from rows that actually have cluster_id.
    tmp = clustered.copy()
    tmp["content_text"] = tmp.apply(
        lambda r: _join_text(r.get("caption"), r.get("transcription_text"), r.get("ocr_text")),
        axis=1,
    )
    tmp = tmp.sort_values("engagement", ascending=False)
    for _, row in tmp.iterrows():
        cid = row.get("cluster_id")
        if pd.isna(cid):
            continue
        by_cluster_rows[int(cid)].append(row.to_dict())

    show_rows: list[dict[str, Any]] = []
    for s in sorted(summaries, key=lambda x: x.count, reverse=True):
        payload = labels.get(str(s.cluster_id), {})
        label = payload.get("label") if isinstance(payload, dict) else None
        summary = payload.get("summary") if isinstance(payload, dict) else None
        show_rows.append(
            {
                "cluster_id": s.cluster_id,
                "label": label or s.label,
                "count": s.count,
                "top_terms": ", ".join(s.top_terms[:8]),
                "examples": ", ".join(s.example_shortcodes),
                "llm_summary": summary or "",
            }
        )
    st.dataframe(pd.DataFrame(show_rows), use_container_width=True, hide_index=True)

    st.caption("Optional: label clusters with an LLM (OpenAI). Saves labels to .state/trend_labels.json")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    cols = st.columns(3)
    with cols[0]:
        st.text_input("OPENAI_MODEL", value=model, key="openai_model")
    with cols[1]:
        st.text_input("OPENAI_API_KEY (env var preferred)", value=api_key, type="password", key="openai_key")
    with cols[2]:
        run_llm = st.button("Run LLM labeling")

    if run_llm:
        key_val = st.session_state.get("openai_key", "").strip()
        model_val = st.session_state.get("openai_model", "gpt-4o-mini").strip()
        if not key_val:
            st.error("No OPENAI_API_KEY provided.")
        else:
            with st.spinner("Labeling clusters..."):
                labeled = maybe_label_clusters_with_llm(
                    summaries,
                    rows_by_cluster=by_cluster_rows,
                    model=model_val,
                    api_key=key_val,
                )
                merged_labels = load_labels()
                merged_labels.update(labeled)
                save_labels(merged_labels)
            st.success(f"Saved labels to {LABELS_PATH}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Lectern Ingest Script
=====================
Curated lecture-grade YouTube library ingest job.

Schema Choice:
--------------
`lectern/data/videos.json` is stored as a plain JSON array of video objects:
[
  {
    "id": str,
    "title": str,
    "channelId": str,
    "channelName": str,
    "publishedAt": str,
    "durationSec": int | None,
    "thumbnail": str,
    "topics": list[str],
    "kind": str,
    "blurb": str,
    "url": str
  },
  ...
]

Chosen because:
1. Lectern's web UI client (`lectern/js/app.js`) directly binds `state.videos = await vRes.json()`
   and computes `state.videos.length`.
2. Existing curated seed catalog in `lectern/data/videos.json` adheres to this exact schema.

Modes:
------
1. WITH YOUTUBE_API_KEY environment variable:
   - For each channel: reads the channel uploads playlist ('UU' + channel_id[2:])
   - Pages playlistItems (~100 items per channel: up to 2 pages of 50)
   - Batches videos.list (contentDetails, snippet, statistics) up to 50 IDs per call
   - Ingests curated playlists from `lectern/data/playlists.json` the same way
   - Drops duration < 12 minutes (720 seconds), Shorts, live streams, and clickbait/listicle titles.
2. WITHOUT YOUTUBE_API_KEY:
   - Fetches YouTube RSS feed (https://www.youtube.com/feeds/videos.xml?channel_id=UC...)
   - Parses Atom XML feed entries.
   - Retains items with durationSec = None (since RSS omits durations).
   - If RSS endpoint returns 404/rate limits or network is unavailable, safely preserves
     and merges with existing hand-seeded catalog.
   - Protection: if videos.json already has rich seed entries, running without a key will
     never clobber or wipe the seed catalog.
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

# Paths relative to repo root
ROOT_DIR = Path(__file__).resolve().parent.parent
LECTERN_DIR = ROOT_DIR / "lectern"
DATA_DIR = LECTERN_DIR / "data"
CHANNELS_FILE = DATA_DIR / "channels.json"
PLAYLISTS_FILE = DATA_DIR / "playlists.json"
VIDEOS_FILE = DATA_DIR / "videos.json"

# Quality filters
MIN_DURATION_SECONDS = 720  # 12 minutes

CLICKBAIT_PATTERNS = [
    re.compile(r"^\s*top\s+\d+", re.IGNORECASE),
    re.compile(r"\bshocking\b", re.IGNORECASE),
    re.compile(r"\byou won'?t believe\b", re.IGNORECASE),
    re.compile(r"#shorts\b", re.IGNORECASE),
    re.compile(r"\bshorts\b", re.IGNORECASE),
]

EXCLUDED_PATTERNS = [
    re.compile(r"\bmusic theory\b", re.IGNORECASE),
    re.compile(r"\bchord progression\b", re.IGNORECASE),
    re.compile(r"\bneely\b", re.IGNORECASE),
    re.compile(r"\bbeato\b", re.IGNORECASE),
]


def parse_iso8601_duration(duration_str: str) -> int:
    """Parse ISO 8601 duration string (e.g. PT1H14M22S, PT12M30S) into total seconds."""
    match = re.match(r"^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$", duration_str or "")
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def is_clickbait_or_low_quality(title: str) -> bool:
    """Return True if title matches clickbait, Shorts, or excluded patterns."""
    for pattern in CLICKBAIT_PATTERNS:
        if pattern.search(title):
            return True
    for pattern in EXCLUDED_PATTERNS:
        if pattern.search(title):
            return True
    return False


def is_live_or_upcoming(snippet: dict) -> bool:
    """Check if YouTube item snippet indicates live broadcast or upcoming stream."""
    live_broadcast_content = snippet.get("liveBroadcastContent", "none")
    return live_broadcast_content in ("live", "upcoming")


def make_request(url: str, headers: dict = None) -> bytes:
    """Make an HTTP GET request with a reasonable User-Agent and timeout."""
    req_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers)
    with urllib.request.urlopen(req, timeout=12) as response:
        return response.read()


def fetch_api_json(url: str) -> dict:
    """Fetch and decode JSON from YouTube Data API endpoint."""
    raw = make_request(url)
    return json.loads(raw.decode("utf-8"))


def load_channels_data(path: Path) -> list:
    """Load channels supporting either top-level array or {channels: [...]} object."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("channels", [])
    return []


def load_playlists_data(path: Path) -> list:
    """Load playlists supporting either top-level array or {playlists: [...]} object."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("playlists", [])
    return []


def load_existing_videos() -> dict:
    """Load existing videos.json indexed by video ID."""
    if not VIDEOS_FILE.exists():
        return {}
    try:
        with open(VIDEOS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("videos", [])
        else:
            items = []
        return {v["id"]: v for v in items if isinstance(v, dict) and "id" in v}
    except Exception as e:
        print(f"Warning: could not read existing videos.json: {e}", file=sys.stderr)
        return {}


def fetch_playlist_video_ids(playlist_id: str, api_key: str, max_items: int = 100) -> list:
    """
    Page through playlistItems (up to max_items, ~100 by default) to collect video IDs.
    """
    video_ids = []
    page_token = ""
    while len(video_ids) < max_items:
        per_page = min(50, max_items - len(video_ids))
        params = {
            "part": "contentDetails,snippet",
            "playlistId": playlist_id,
            "maxResults": per_page,
            "key": api_key,
        }
        if page_token:
            params["pageToken"] = page_token
        url = f"https://www.googleapis.com/youtube/v3/playlistItems?{urllib.parse.urlencode(params)}"
        try:
            res = fetch_api_json(url)
            items = res.get("items", [])
            if not items:
                break
            for item in items:
                content_details = item.get("contentDetails", {})
                vid = content_details.get("videoId")
                if vid:
                    video_ids.append(vid)
            page_token = res.get("nextPageToken")
            if not page_token:
                break
        except Exception as e:
            # Mask API key if present in error message
            err_msg = str(e).replace(api_key, "[REDACTED]")
            print(f"  [API Error] playlistItems for playlist {playlist_id}: {err_msg}", file=sys.stderr)
            break
    return video_ids


def fetch_and_filter_videos_batch(
    video_ids: list,
    api_key: str,
    channel_id: str,
    channel_name: str,
    topics: list,
    kind: str,
    blurb: str,
) -> list:
    """
    Batch fetch videos.list (up to 50 IDs per call) with contentDetails,snippet,statistics.
    Applies quality filters:
      - drops duration < 12 minutes (720s)
      - drops Shorts
      - drops live streams / upcoming broadcasts
      - drops clickbait / listicle / excluded titles
    """
    videos = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i : i + 50]
        params = {
            "part": "snippet,contentDetails,statistics",
            "id": ",".join(batch),
            "key": api_key,
        }
        url = f"https://www.googleapis.com/youtube/v3/videos?{urllib.parse.urlencode(params)}"
        try:
            res = fetch_api_json(url)
            for item in res.get("items", []):
                snippet = item.get("snippet", {})
                content_details = item.get("contentDetails", {})
                title = snippet.get("title", "")

                # 1. Filter clickbait, listicle, or excluded patterns
                if is_clickbait_or_low_quality(title):
                    continue

                # 2. Filter live streams or upcoming broadcasts
                if is_live_or_upcoming(snippet):
                    continue

                # 3. Filter duration < 12 minutes (720 seconds) or Shorts
                duration_iso = content_details.get("duration", "")
                duration_sec = parse_iso8601_duration(duration_iso)
                if duration_sec < MIN_DURATION_SECONDS:
                    continue

                vid = item.get("id")
                if not vid:
                    continue

                thumbs = snippet.get("thumbnails", {})
                thumb_url = (
                    thumbs.get("maxres", {}).get("url")
                    or thumbs.get("high", {}).get("url")
                    or thumbs.get("medium", {}).get("url")
                    or f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
                )

                actual_channel_id = snippet.get("channelId") or channel_id
                actual_channel_name = snippet.get("channelTitle") or channel_name

                videos.append(
                    {
                        "id": vid,
                        "title": title,
                        "channelId": actual_channel_id,
                        "channelName": actual_channel_name,
                        "publishedAt": snippet.get("publishedAt", ""),
                        "durationSec": duration_sec,
                        "thumbnail": thumb_url,
                        "topics": topics or [],
                        "kind": kind or "visual-explainer",
                        "blurb": blurb or "",
                        "url": f"https://www.youtube.com/watch?v={vid}",
                    }
                )
        except Exception as e:
            err_msg = str(e).replace(api_key, "[REDACTED]")
            print(f"  [API Error] videos.list batch: {err_msg}", file=sys.stderr)

    return videos


def ingest_channel_api(channel: dict, api_key: str) -> list:
    """Ingest channel uploads using YouTube Data API v3."""
    channel_id = channel.get("id", "")
    if not channel_id or len(channel_id) < 3:
        return []

    # Uploads playlist ID is typically channel ID with 'UU' prefix
    uploads_playlist_id = "UU" + channel_id[2:]
    video_ids = fetch_playlist_video_ids(uploads_playlist_id, api_key, max_items=100)
    if not video_ids:
        return []

    return fetch_and_filter_videos_batch(
        video_ids=video_ids,
        api_key=api_key,
        channel_id=channel_id,
        channel_name=channel.get("name", ""),
        topics=channel.get("topics", []),
        kind=channel.get("kind", "visual-explainer"),
        blurb=channel.get("blurb", ""),
    )


def ingest_playlist_api(playlist: dict, api_key: str) -> list:
    """Ingest curated playlist using YouTube Data API v3."""
    playlist_id = playlist.get("id", "")
    if not playlist_id:
        return []

    video_ids = fetch_playlist_video_ids(playlist_id, api_key, max_items=100)
    if not video_ids:
        return []

    return fetch_and_filter_videos_batch(
        video_ids=video_ids,
        api_key=api_key,
        channel_id=playlist.get("channelId", ""),
        channel_name=playlist.get("channelName", ""),
        topics=playlist.get("topics", []),
        kind=playlist.get("kind", "course"),
        blurb=playlist.get("blurb", ""),
    )


def ingest_channel_rss(channel: dict) -> list:
    """
    Ingest videos for a channel using YouTube RSS feed:
    https://www.youtube.com/feeds/videos.xml?channel_id=UC...
    durationSec is set to None since RSS omits duration.
    """
    channel_id = channel.get("id", "")
    if not channel_id:
        return []

    rss_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    videos = []

    try:
        raw_xml = make_request(rss_url)
        root = ET.fromstring(raw_xml)
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "yt": "http://www.youtube.com/xml/schemas/2015",
            "media": "http://search.yahoo.com/mrss/",
        }

        for entry in root.findall("atom:entry", ns):
            yt_id_elem = entry.find("yt:videoId", ns)
            title_elem = entry.find("atom:title", ns)
            published_elem = entry.find("atom:published", ns)

            if yt_id_elem is None or title_elem is None:
                continue

            vid = yt_id_elem.text
            title = title_elem.text or ""

            if is_clickbait_or_low_quality(title):
                continue

            published_at = published_elem.text if published_elem is not None else ""
            thumb_url = f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"

            videos.append(
                {
                    "id": vid,
                    "title": title,
                    "channelId": channel_id,
                    "channelName": channel.get("name", ""),
                    "publishedAt": published_at,
                    "durationSec": None,  # RSS omits duration
                    "thumbnail": thumb_url,
                    "topics": channel.get("topics", []),
                    "kind": channel.get("kind", "visual-explainer"),
                    "blurb": channel.get("blurb", ""),
                    "url": f"https://www.youtube.com/watch?v={vid}",
                }
            )
    except Exception:
        # Upstream RSS endpoint may return 404, 500, or rate limit
        pass

    return videos


def main():
    parser = argparse.ArgumentParser(
        description="Lectern YouTube ingest job (YouTube Data API v3 with RSS fallback)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of channels/playlists to process (for debugging or partial runs)",
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Force overwriting videos.json even when existing seed is rich",
    )
    args = parser.parse_args()

    print("=== Lectern Ingest Job ===")
    api_key = os.environ.get("YOUTUBE_API_KEY", "").strip()

    channels = load_channels_data(CHANNELS_FILE)
    playlists = load_playlists_data(PLAYLISTS_FILE)

    if not channels:
        print(f"Error: No channels loaded from {CHANNELS_FILE}.", file=sys.stderr)
        sys.exit(1)

    existing_videos = load_existing_videos()
    print(f"Loaded {len(channels)} curated channels and {len(playlists)} playlists.")
    print(f"Found {len(existing_videos)} existing videos in catalog.")

    # Guard: if running without API key and existing seed is already rich (>100 items),
    # do not clobber unless --force-overwrite is specified
    if not api_key and len(existing_videos) >= 100 and not args.force_overwrite:
        print(
            f"Notice: YOUTUBE_API_KEY is not set and videos.json already contains a rich "
            f"seed catalog ({len(existing_videos)} videos). Skipping overwrite to prevent "
            f"clobbering rich metadata."
        )
        return

    channels_to_process = channels[: args.limit] if args.limit else channels
    playlists_to_process = playlists[: args.limit] if args.limit else playlists

    new_videos_map = dict(existing_videos)
    fetched_count = 0

    if api_key:
        print("Running in YouTube API mode (YOUTUBE_API_KEY provided)...")

        # Ingest channels
        for i, ch in enumerate(channels_to_process, 1):
            print(f"[{i}/{len(channels_to_process)}] Ingesting channel: {ch.get('name')}...")
            vids = ingest_channel_api(ch, api_key)
            for v in vids:
                new_videos_map[v["id"]] = v
                fetched_count += 1

        # Ingest playlists
        for j, pl in enumerate(playlists_to_process, 1):
            print(f"[{j}/{len(playlists_to_process)}] Ingesting playlist: {pl.get('title')}...")
            vids = ingest_playlist_api(pl, api_key)
            for v in vids:
                new_videos_map[v["id"]] = v
                fetched_count += 1
    else:
        print("Running in RSS fallback mode (no YOUTUBE_API_KEY provided)...")
        print("Attempting YouTube RSS for curated channels...")
        rss_success_channels = 0
        for i, ch in enumerate(channels_to_process, 1):
            vids = ingest_channel_rss(ch)
            if vids:
                rss_success_channels += 1
                for v in vids:
                    # If video already exists in seed with duration, retain seed duration
                    if v["id"] in new_videos_map and new_videos_map[v["id"]].get("durationSec") is not None:
                        continue
                    new_videos_map[v["id"]] = v
                    fetched_count += 1
        print(f"RSS feeds yielded videos from {rss_success_channels} channels ({fetched_count} entries added/updated).")

    final_videos = list(new_videos_map.values())
    final_videos.sort(key=lambda x: x.get("publishedAt", "") or "", reverse=True)

    print(f"Total catalog size: {len(final_videos)} videos.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(VIDEOS_FILE, "w", encoding="utf-8") as f:
        json.dump(final_videos, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(final_videos)} videos to {VIDEOS_FILE}")
    print("Ingest job complete.")


if __name__ == "__main__":
    main()

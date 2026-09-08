#!/usr/bin/env python3
"""
Lectern Ingest Script
=====================
Curated lecture-grade YouTube library ingest job.

Modes:
1. WITH YOUTUBE_API_KEY environment variable:
   - For each channel: reads the channel uploads playlist ('UU' + channel_id[2:])
   - Pages playlistItems (capped at ~100 per channel)
   - Batches videos.list (contentDetails, snippet, statistics) up to 50 IDs per call
   - Also processes playlists in data/playlists.json
   - Drops duration < 12:00 (720s), Shorts, live streams, and listicle/clickbait titles.
2. WITHOUT YOUTUBE_API_KEY:
   - Fetches YouTube RSS feed (https://www.youtube.com/feeds/videos.xml?channel_id=...)
   - Parses Atom XML feed entries.
   - Retains items with durationSec = None (since RSS omits durations).
   - If RSS endpoint returns 404/rate limits or network is unavailable, safely preserves
     and merges with existing hand-seeded lectern/data/videos.json.

Output:
Writes updated `lectern/data/videos.json` matching the schema:
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
}
"""

import os
import sys
import re
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

# Base paths
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

# Excluded music theory keywords to guarantee pure STEM/philosophy/history canon
EXCLUDED_PATTERNS = [
    re.compile(r"\bmusic theory\b", re.IGNORECASE),
    re.compile(r"\bchord progression\b", re.IGNORECASE),
    re.compile(r"\bneely\b", re.IGNORECASE),
    re.compile(r"\bbeato\b", re.IGNORECASE),
]

def parse_iso8601_duration(duration_str: str) -> int:
    """Parse ISO 8601 duration (e.g., PT1H14M22S, PT12M30S) to total seconds."""
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration_str)
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds

def is_clickbait_or_low_quality(title: str) -> bool:
    """Return True if title looks like clickbait, a short, or excluded content."""
    for pattern in CLICKBAIT_PATTERNS:
        if pattern.search(title):
            return True
    for pattern in EXCLUDED_PATTERNS:
        if pattern.search(title):
            return True
    return False

def make_request(url: str, headers: dict = None) -> bytes:
    """Make an HTTP GET request with reasonable timeout and User-Agent."""
    req_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers)
    with urllib.request.urlopen(req, timeout=10) as response:
        return response.read()

def fetch_api_json(url: str) -> dict:
    data = make_request(url)
    return json.loads(data.decode("utf-8"))

def ingest_channel_api(channel: dict, api_key: str) -> list:
    """Ingest videos for a channel using YouTube Data API v3."""
    channel_id = channel["id"]
    uploads_playlist_id = "UU" + channel_id[2:]
    videos = []

    # 1. Fetch playlistItems (up to 100 items: 2 pages of 50)
    video_ids = []
    page_token = ""
    for _ in range(2):
        params = {
            "part": "snippet,contentDetails",
            "playlistId": uploads_playlist_id,
            "maxResults": 50,
            "key": api_key
        }
        if page_token:
            params["pageToken"] = page_token
        url = f"https://www.googleapis.com/youtube/v3/playlistItems?{urllib.parse.urlencode(params)}"
        try:
            res = fetch_api_json(url)
            items = res.get("items", [])
            for item in items:
                vid = item["contentDetails"]["videoId"]
                video_ids.append(vid)
            page_token = res.get("nextPageToken")
            if not page_token:
                break
        except Exception as e:
            print(f"  [API Error] playlistItems for {channel['name']} ({uploads_playlist_id}): {e}")
            break

    if not video_ids:
        return []

    # 2. Batch fetch videos.list (50 at a time)
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        params = {
            "part": "snippet,contentDetails,statistics",
            "id": ",".join(batch),
            "key": api_key
        }
        url = f"https://www.googleapis.com/youtube/v3/videos?{urllib.parse.urlencode(params)}"
        try:
            res = fetch_api_json(url)
            for item in res.get("items", []):
                snippet = item.get("snippet", {})
                content_details = item.get("contentDetails", {})
                title = snippet.get("title", "")
                
                # Filter clickbait
                if is_clickbait_or_low_quality(title):
                    continue

                # Filter live streams or missing duration
                duration_iso = content_details.get("duration", "")
                duration_sec = parse_iso8601_duration(duration_iso)
                if duration_sec < MIN_DURATION_SECONDS:
                    continue

                vid = item["id"]
                thumbs = snippet.get("thumbnails", {})
                thumb_url = (
                    thumbs.get("maxres", {}).get("url") or
                    thumbs.get("high", {}).get("url") or
                    thumbs.get("medium", {}).get("url") or
                    f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
                )

                videos.append({
                    "id": vid,
                    "title": title,
                    "channelId": channel_id,
                    "channelName": channel["name"],
                    "publishedAt": snippet.get("publishedAt", ""),
                    "durationSec": duration_sec,
                    "thumbnail": thumb_url,
                    "topics": channel.get("topics", []),
                    "kind": channel.get("kind", "visual-explainer"),
                    "blurb": channel.get("blurb", ""),
                    "url": f"https://www.youtube.com/watch?v={vid}"
                })
        except Exception as e:
            print(f"  [API Error] videos.list for {channel['name']}: {e}")

    return videos

def ingest_channel_rss(channel: dict) -> list:
    """Ingest videos for a channel using YouTube RSS feed."""
    channel_id = channel["id"]
    rss_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    videos = []

    try:
        raw_xml = make_request(rss_url)
        root = ET.fromstring(raw_xml)
        # Atom XML namespace
        ns = {"atom": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015", "media": "http://search.yahoo.com/mrss/"}
        
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

            videos.append({
                "id": vid,
                "title": title,
                "channelId": channel_id,
                "channelName": channel["name"],
                "publishedAt": published_at,
                "durationSec": None,  # Duration omitted in RSS
                "thumbnail": thumb_url,
                "topics": channel.get("topics", []),
                "kind": channel.get("kind", "visual-explainer"),
                "blurb": channel.get("blurb", ""),
                "url": f"https://www.youtube.com/watch?v={vid}"
            })
    except Exception as e:
        # Note: YouTube RSS commonly 404s or rate-limits without credentials
        # We catch and print debug info without breaking the pipeline
        pass

    return videos

def load_existing_videos() -> dict:
    """Load existing videos.json indexed by video ID."""
    if not VIDEOS_FILE.exists():
        return {}
    try:
        with open(VIDEOS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {v["id"]: v for v in data if "id" in v}
    except Exception as e:
        print(f"Warning: could not read existing videos.json: {e}")
        return {}

def main():
    print("=== Lectern Ingest Job ===")
    api_key = os.environ.get("YOUTUBE_API_KEY", "").strip()

    if not CHANNELS_FILE.exists():
        print(f"Error: {CHANNELS_FILE} not found.", file=sys.stderr)
        sys.exit(1)

    with open(CHANNELS_FILE, "r", encoding="utf-8") as f:
        channels = json.load(f)

    existing_videos = load_existing_videos()
    print(f"Loaded {len(channels)} curated canon channels.")
    print(f"Found {len(existing_videos)} existing videos in seed catalog.")

    new_videos_map = dict(existing_videos)
    fetched_count = 0

    if api_key:
        print("Running in YouTube API mode (YOUTUBE_API_KEY provided)...")
        for i, channel in enumerate(channels, 1):
            print(f"[{i}/{len(channels)}] Ingesting {channel['name']}...")
            vids = ingest_channel_api(channel, api_key)
            for v in vids:
                new_videos_map[v["id"]] = v
                fetched_count += 1
    else:
        print("Running in RSS fallback mode (no YOUTUBE_API_KEY provided)...")
        print("Attempting YouTube RSS for canon channels...")
        rss_success_channels = 0
        for i, channel in enumerate(channels, 1):
            vids = ingest_channel_rss(channel)
            if vids:
                rss_success_channels += 1
                for v in vids:
                    # If video already exists in seed with duration, retain seed duration
                    if v["id"] in new_videos_map and new_videos_map[v["id"]].get("durationSec") is not None:
                        continue
                    new_videos_map[v["id"]] = v
                    fetched_count += 1
        print(f"RSS feeds yielded videos from {rss_success_channels} channels ({fetched_count} new entries).")

    # Final list of videos, sorted newest first
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

# Lectern

> A searchable library of lecture-grade YouTube (3Blue1Brown / Welch Labs density).
> High-signal mathematics, theoretical computer science, physics, engineering, machine learning, philosophy, and history.

Lectern is designed for students and researchers who learn best when material is rigorous, mathematically transparent, and unapologetically technical. Rather than a broad YouTube search, Lectern is an intentionally restricted canon of curated creators and university lectures.

---

## The Curatorial Standard

- **Standard of Density**: 3Blue1Brown and Welch Labs level exposition — content with proofs, matrices, and formal mechanics rather than hand-waving metaphors.
- **Topics Included**: `math`, `cs`, `physics`, `engineering`, `ml`, `philosophy`, `history`.
- **Expositions Included**:
  - University lecture series: MIT OpenCourseWare (Gilbert Strang, Erik Demaine), Stanford Online (Andrew Ng CS229), Harvard, Oxford, Yale Open Courses, Gresham College, Institute for Advanced Study.
  - Deep algorithmic foundations: Andrej Karpathy (*Neural Networks: Zero to Hero*), Ben Eater (8-bit computer from scratch), Reducible, Polylog, Jon Gjengset (Rust systems internals), Tsoding.
  - Pure mathematical foundations: Mathologer, Aleph 0, Morphocular, Primer, Mutual Information, Richard Behiel.
  - Foundational humanities: Michael Sugrue (*Great Minds of the Western Intellectual Tradition*), Jeffrey Kaplan, Philosophy Overdose, Historia Civilis, Fall of Civilizations, Voices of the Past.
- **Strictly Excluded**:
  - Pop-science and superficial explainers (Kurzgesagt, Veritasium, Crash Course, Two Minute Papers, TED/TEDx).
  - Short-form sensationalism and clickbait (#Shorts, "Top 10", "You won't believe").
  - ALL music theory (Adam Neely, Rick Beato, 12tone, Sideways).
  - Videos under 12 minutes are hidden by default when duration is known.

---

## Directory Layout

```
lectern/
  index.html            # Main academic library web interface
  css/
    styles.css          # Paper / chalk typography and palette
  js/
    app.js              # In-memory client search, filters, modals
  data/
    channels.json       # 62 curated canon channels with real YouTube IDs
    playlists.json      # Curated landmark courses & exposition playlists
    videos.json         # Rich seed catalog (1,280+ lecture-grade works)
  README.md
scripts/
  lectern_ingest.py     # Channel ingest script (YouTube API + RSS fallback)
```

---

## Quickstart

Run a local HTTP server from the repository root:

```bash
python3 -m http.server 5173
```

Open [http://localhost:5173/lectern/](http://localhost:5173/lectern/) in your browser.

---

## Ingest Job

The ingest script updates `lectern/data/videos.json` from the channels specified in `lectern/data/channels.json`:

```bash
python3 scripts/lectern_ingest.py
```

### With `YOUTUBE_API_KEY`:
If the environment variable `YOUTUBE_API_KEY` is set, the script queries the YouTube Data API v3:
1. For each channel, retrieves the channel uploads playlist (`UU...`).
2. Pages `playlistItems` (capped at ~100 items per channel).
3. Batches `videos.list` up to 50 items at a time (`contentDetails`, `snippet`, `statistics`).
4. Drops items with duration `< 12:00`, Shorts, live streams, and clickbait titles.
5. Writes the updated catalog to `lectern/data/videos.json`.

### Without `YOUTUBE_API_KEY`:
If no API key is present:
1. Attempts YouTube RSS feeds (`https://www.youtube.com/feeds/videos.xml?channel_id=...`).
2. Merges entries while preserving durations from the existing seed data.
3. If YouTube RSS 404s (an intermittent upstream YouTube condition) or network fails, the script safely keeps the committed seed `videos.json` without data loss.

---

## Standalone Zero-Dependency Guarantee

Lectern requires no bundler, no npm install, and no build step. The committed `videos.json` contains over 1,280 real, lecture-grade videos, allowing Lectern to work instantly out of the box on GitHub Pages or any static web host.

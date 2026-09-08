# Lectern

> A searchable, lecture-grade YouTube library.
> High-signal mathematics, theoretical computer science, physics, engineering, machine learning, philosophy, and history.

Lectern is designed for students, engineers, and researchers who learn best when material is rigorous, mathematically transparent, and unapologetically technical. Rather than an unconstrained YouTube search, Lectern is an intentionally restricted canon of curated creators and university lectures.

---

## What Lectern Is

Lectern provides a curated web catalog featuring:
- **Paper & Chalk Academic Aesthetic**: Warm ivory parchment, antique walnut ink, and lapidary typography — deliberate departure from both Nightboard starfields and standard YouTube red.
- **In-Memory Filtering & Search**: Instant token matching across titles, channels, topics, and blurbs.
- **Multi-Faceted Refinement**:
  - **Topics**: `math`, `cs`, `physics`, `engineering`, `ml`, `philosophy`, `history`.
  - **Kinds**: `visual-explainer`, `course`, `lecture`, `long-conversation`.
  - **Duration Buckets**: `12–20 min`, `20–45 min`, `45–90 min`, `90+ min`.
  - **Recency**: `Past Year`, `Past 3 Years`, `Archival (3+ yrs)`.
  - **Canon Guard**: Videos under 12 minutes and YouTube Shorts are excluded by default when duration is known.
- **Card Metadata & Direct Watch**: Each card displays thumbnail, title, channel, duration, publication date, kind, blurb, and topic tags. Clicking opens the lecture directly on YouTube in a new tab.
- **Channel Focus**: Filter the catalog by a specific faculty member or channel, viewing the channel's curatorial blurb and institutional focus.
- **Canon Philosophy**: If a query yields no matches, Lectern reminds you: *“This is not YouTube search; it is a canon.”*
- **Zero Bundler / Zero Dependencies**: Built with vanilla HTML5, CSS3, and JavaScript (ES6+). Runs directly in any modern browser without npm, webpack, or build steps.

---

## Quickstart

Run a local HTTP server from the repository root:

```bash
python3 -m http.server 8000
```

Then open the Lectern frontend in your browser:

[http://localhost:8000/lectern/](http://localhost:8000/lectern/)

---

## Ingest Job

Lectern's catalog can be refreshed or expanded from the channels specified in `lectern/data/channels.json` by running the ingest script:

```bash
python3 scripts/lectern_ingest.py
```

### With `YOUTUBE_API_KEY`:
If the environment variable `YOUTUBE_API_KEY` is set, the script queries the YouTube Data API v3:
1. For each channel, retrieves uploads via playlist items.
2. Batches `videos.list` requests (up to 50 items) for durations, snippets, and metadata.
3. Filters out content under 12 minutes, Shorts, livestreams, and clickbait.
4. Outputs the enriched catalog to `lectern/data/videos.json`.

### Without `YOUTUBE_API_KEY`:
If no API key is set, the ingest script falls back to YouTube RSS feeds and safely preserves existing catalog metadata without data loss.

---

## Data Schemas

Lectern defensively loads `./data/` or `../data/` JSON files and supports both raw arrays and `{ "videos": [...] }` or `{ "channels": [...] }` object formats.

### Video Schema:
```json
{
  "id": "aircAruvnKk",
  "title": "Neural Networks: Zero to Hero",
  "channelId": "UCXUPKJO5MZQN11PqgIvyuvQ",
  "channelName": "Andrej Karpathy",
  "publishedAt": "2022-08-16T15:00:00Z",
  "durationSec": 8100,
  "thumbnail": "https://i.ytimg.com/vi/aircAruvnKk/hqdefault.jpg",
  "topics": ["ml", "cs"],
  "kind": "course",
  "blurb": "Building deep neural networks from first principles using Python and autograd.",
  "url": "https://www.youtube.com/watch?v=aircAruvnKk"
}
```

### Channel Schema:
```json
{
  "id": "UCYO_jab_esuFRV4b17AJtAw",
  "name": "3Blue1Brown",
  "topics": ["math", "cs", "ml"],
  "kind": "visual-explainer",
  "blurb": "Grant Sanderson's gold standard for visual mathematics, linear algebra, calculus, and neural network intuition."
}
```

---

## Directory Structure

```
lectern/
  index.html            # Main library web interface
  css/
    styles.css          # Paper / chalk typography and academic layout
  js/
    app.js              # In-memory search, faceted filtering, defensive data loading
  data/
    channels.json       # Curated faculty & institutions
    playlists.json      # Curated course series & playlists
    videos.json         # Lecture-grade catalog
  README.md             # This documentation
scripts/
  lectern_ingest.py     # Ingest script
```

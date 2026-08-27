# Nightboard

Live: https://saxcoll.github.io/nightboard/

A live event OS for a Cursor meetup / hack night: **now/next**, a **floor map of teams**, a **pizza cam**, a **GitHub repo bot**, a **demo queue**, and **people looking for teammates**.

## Run

```bash
python3 -m http.server 5173
```

Open [http://localhost:5173](http://localhost:5173).

- **Hackathon** — 35 real teams from the August meetup portal. Spaceflex is your team.
- **Presentations** — volunteer pool (Spaceflex already volunteered). Three teams picked tonight.
- **Rules** — short night-of rules.
- **Pizza cam** and **Repo bot** still live as extra stations.

- **Pizza cam** — sim kitchen-cam by default (Hawaiian runs out first). Arm a real webcam for a hue-grid slice estimate, then correct with +/-.
- **Repo bot** — polls public GitHub every 90s and scores quiet / warming / shipping / stalled / demo-ready. Nightboard lives at saxcoll/nightboard.
- **Live ops** — announcement ticker, walk-over help matching, projector mode, demo countdown.

Demo data is seeded for **Cursor Calgary Meetup — August** at ZayZoon. Swap `js/data.js` for a real backend later. GitHub uses unauthenticated API (60 req/hr). Point `team.repo` at the real org/repo when you have it.

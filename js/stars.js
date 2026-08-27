/** Sparse canvas starfield — deep void, slow twinkle, faint drift. */

const VOID = "#05060a";
const STAR = [226, 232, 245];

function initStars(canvas) {
  const ctx = canvas.getContext("2d", { alpha: false });
  if (!ctx) return;

  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let stars = [];
  let w = 0;
  let h = 0;
  let raf = 0;
  let t0 = performance.now();

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = window.innerWidth;
    h = window.innerHeight;
    canvas.width = Math.max(1, Math.floor(w * dpr));
    canvas.height = Math.max(1, Math.floor(h * dpr));
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    seed();
  }

  function seed() {
    const area = w * h;
    const count = Math.max(48, Math.min(160, Math.round(area / 14000)));
    stars = [];
    for (let i = 0; i < count; i++) {
      const layer = Math.random();
      const bright = Math.random() > 0.92;
      stars.push({
        x: Math.random() * w,
        y: Math.random() * h,
        r: bright ? 1.2 : layer > 0.72 ? 0.85 : layer > 0.38 ? 0.55 : 0.4,
        base: bright ? 0.62 : 0.16 + layer * 0.42,
        twinkle: bright ? 0.12 : 0.06 + Math.random() * 0.16,
        phase: Math.random() * Math.PI * 2,
        period: 2200 + Math.random() * 4800,
        drift: 2.5 + layer * 6,
      });
    }
  }

  function paintHorizon() {
    const g = ctx.createLinearGradient(0, h * 0.7, 0, h);
    g.addColorStop(0, "rgba(5, 6, 10, 0)");
    g.addColorStop(1, "rgba(10, 14, 24, 0.42)");
    ctx.fillStyle = g;
    ctx.fillRect(0, h * 0.7, w, h * 0.3);
  }

  function draw(now) {
    const t = now - t0;
    ctx.fillStyle = VOID;
    ctx.fillRect(0, 0, w, h);

    for (const s of stars) {
      const pulse = reduced ? 0 : Math.sin((t / s.period) * Math.PI * 2 + s.phase) * s.twinkle;
      const a = Math.min(0.92, Math.max(0.06, s.base + pulse));
      const dx = reduced ? 0 : ((t / 90000) * s.drift) % (w + 16);
      const x = (s.x + dx) % (w + 8);
      ctx.beginPath();
      ctx.fillStyle = `rgba(${STAR[0]}, ${STAR[1]}, ${STAR[2]}, ${a})`;
      ctx.arc(x < 0 ? x + w : x, s.y, s.r, 0, Math.PI * 2);
      ctx.fill();
    }

    paintHorizon();
    if (!reduced) raf = requestAnimationFrame(draw);
  }

  resize();
  window.addEventListener("resize", resize);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      cancelAnimationFrame(raf);
      raf = 0;
    } else if (!raf) {
      t0 = performance.now() - (performance.now() - t0);
      raf = requestAnimationFrame(draw);
    }
  });

  if (reduced) {
    draw(t0);
    return;
  }
  raf = requestAnimationFrame(draw);
}

const canvas = document.getElementById("starfield");
if (canvas) initStars(canvas);

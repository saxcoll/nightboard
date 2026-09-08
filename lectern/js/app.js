/**
 * Lectern — Curated Lecture-Grade YouTube Library
 * Vanilla JavaScript application: in-memory search, multi-faceted filtering,
 * defensive data loading, channel focus banners, and zero external dependencies.
 */

(function () {
  "use strict";

  // Application State
  const state = {
    videos: [],
    channels: [],
    playlists: [],
    channelsMap: new Map(), // channelId -> channel object
    searchQuery: "",
    selectedTopic: "all",
    selectedKind: "all",
    selectedDuration: "all",
    selectedRecency: "all",
    selectedSort: "relevant",
    selectedChannelId: null,
    hideShortsAndUnder12m: true, // Default hide under 12 min and Shorts when duration known
    currentView: "catalog", // 'catalog' | 'channels' | 'playlists'
  };

  // DOM Elements
  const searchInput = document.getElementById("library-search");
  const clearSearchBtn = document.getElementById("clear-search-btn");
  const topicChips = document.getElementById("topic-chips");
  const kindChips = document.getElementById("kind-chips");
  const durationChips = document.getElementById("duration-chips");
  const recencyChips = document.getElementById("recency-chips");
  const sortChips = document.getElementById("sort-chips");
  const toggleShortsGuardBtn = document.getElementById("toggle-shorts-guard");

  const activeQueryBar = document.getElementById("active-query-bar");
  const resultsCountText = document.getElementById("results-count-text");
  const appliedTags = document.getElementById("applied-tags");
  const resetFiltersBtn = document.getElementById("reset-filters-btn");
  const emptyResetBtn = document.getElementById("empty-reset-btn");

  // Channel Banner Elements
  const channelFilterBanner = document.getElementById("channel-filter-banner");
  const bannerChannelKind = document.getElementById("banner-channel-kind");
  const bannerChannelName = document.getElementById("banner-channel-name");
  const bannerChannelTopics = document.getElementById("banner-channel-topics");
  const bannerChannelBlurb = document.getElementById("banner-channel-blurb");
  const bannerChannelCount = document.getElementById("banner-channel-count");
  const bannerChannelYt = document.getElementById("banner-channel-yt");
  const bannerChannelClear = document.getElementById("banner-channel-clear");

  // View Sections
  const catalogSection = document.getElementById("catalog-section");
  const channelsSection = document.getElementById("channels-section");
  const coursesSection = document.getElementById("courses-section");
  const videoGrid = document.getElementById("video-grid");
  const facultyGrid = document.getElementById("faculty-grid");
  const coursesGrid = document.getElementById("courses-grid");
  const emptyState = document.getElementById("empty-state");

  // View Toggles
  const viewModeCatalog = document.getElementById("view-mode-catalog");
  const viewModeChannels = document.getElementById("view-mode-channels");
  const viewModePlaylists = document.getElementById("view-mode-playlists");

  // Header Counters
  const statVideos = document.getElementById("stat-videos");
  const statChannels = document.getElementById("stat-channels");

  /**
   * Defensive JSON Loader
   * Tries candidate paths (./data, ../data, data, /lectern/data)
   */
  async function fetchJsonDefensive(filename) {
    const candidatePaths = [
      `./data/${filename}`,
      `../data/${filename}`,
      `data/${filename}`,
      `/lectern/data/${filename}`,
    ];

    for (const path of candidatePaths) {
      try {
        const response = await fetch(path);
        if (response.ok) {
          return await response.json();
        }
      } catch (err) {
        // Continue to next candidate path
      }
    }
    return null;
  }

  /**
   * Normalize video collection (supports raw array or { videos: [...] })
   */
  function extractVideos(data) {
    if (!data) return [];
    if (Array.isArray(data)) return data;
    if (data && Array.isArray(data.videos)) return data.videos;
    return [];
  }

  /**
   * Normalize channels collection (supports raw array or { channels: [...] })
   */
  function extractChannels(data) {
    if (!data) return [];
    if (Array.isArray(data)) return data;
    if (data && Array.isArray(data.channels)) return data.channels;
    return [];
  }

  /**
   * Normalize playlists collection (supports raw array or { playlists: [...] })
   */
  function extractPlaylists(data) {
    if (!data) return [];
    if (Array.isArray(data)) return data;
    if (data && Array.isArray(data.playlists)) return data.playlists;
    return [];
  }

  /**
   * Format duration in seconds to "MM:SS" or "H:MM:SS"
   */
  function formatDuration(seconds) {
    if (seconds == null || isNaN(seconds)) return "—";
    const totalSec = Math.round(Number(seconds));
    const h = Math.floor(totalSec / 3600);
    const m = Math.floor((totalSec % 3600) / 60);
    const s = Math.floor(totalSec % 60);
    if (h > 0) {
      return `${h}:${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
    }
    return `${m}:${s.toString().padStart(2, "0")}`;
  }

  /**
   * Format ISO date string into readable academic format: "Sep 2024"
   */
  function formatDate(isoStr) {
    if (!isoStr) return "";
    try {
      const d = new Date(isoStr);
      if (isNaN(d.getTime())) return "";
      return d.toLocaleDateString("en-US", { year: "numeric", month: "short" });
    } catch {
      return "";
    }
  }

  /**
   * Escape HTML to prevent injection
   */
  function escapeHTML(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  /**
   * Detect short-form video
   */
  function isShortVideo(video) {
    if (video.durationSec != null && video.durationSec < 60) return true;
    if (video.kind === "short") return true;
    const title = (video.title || "").toLowerCase();
    if (title.includes("#shorts") || title.includes("#short")) return true;
    const url = (video.url || "").toLowerCase();
    if (url.includes("/shorts/")) return true;
    return false;
  }

  /**
   * Match duration bucket: 12-20, 20-45, 45-90, 90+
   */
  function matchesDurationBucket(sec, bucket) {
    if (bucket === "all") return true;
    if (sec == null) return false; // Specific duration filter requires known duration
    const min = sec / 60;
    if (bucket === "12-20") return min >= 12 && min < 20;
    if (bucket === "20-45") return min >= 20 && min < 45;
    if (bucket === "45-90") return min >= 45 && min < 90;
    if (bucket === "90+") return min >= 90;
    return true;
  }

  /**
   * Match recency bucket: past-1, past-3, classic
   */
  function matchesRecencyBucket(publishedAt, bucket) {
    if (bucket === "all") return true;
    if (!publishedAt) return false;
    const pubDate = new Date(publishedAt).getTime();
    if (isNaN(pubDate)) return false;
    const now = Date.now();
    const oneYearMs = 365.25 * 24 * 3600 * 1000;
    const diffMs = now - pubDate;

    if (bucket === "past-1") return diffMs <= oneYearMs;
    if (bucket === "past-3") return diffMs <= 3 * oneYearMs;
    if (bucket === "classic") return diffMs > 3 * oneYearMs;
    return true;
  }

  /**
   * In-memory search scoring
   */
  function scoreVideo(video, queryTokens) {
    if (queryTokens.length === 0) return 1;

    const titleLower = (video.title || "").toLowerCase();
    const channelLower = (video.channelName || "").toLowerCase();
    const blurbLower = (video.blurb || "").toLowerCase();
    const topicsLower = (video.topics || []).join(" ").toLowerCase();

    let score = 0;
    for (const token of queryTokens) {
      if (titleLower.includes(token)) {
        score += 12;
      } else if (channelLower.includes(token)) {
        score += 8;
      } else if (topicsLower.includes(token)) {
        score += 5;
      } else if (blurbLower.includes(token)) {
        score += 3;
      } else {
        return 0; // All tokens must match somewhere (AND search)
      }
    }
    return score;
  }

  /**
   * Filter and sort videos
   */
  function getFilteredVideos() {
    const rawQuery = state.searchQuery.trim().toLowerCase();
    const tokens = rawQuery.length > 0 ? rawQuery.split(/\s+/).filter(Boolean) : [];

    let list = state.videos.filter((v) => {
      // 1. Default hide under 12 min and Shorts when duration known
      if (state.hideShortsAndUnder12m) {
        if (isShortVideo(v)) return false;
        if (v.durationSec != null && v.durationSec < 720) return false;
      }

      // 2. Channel focus filter
      if (state.selectedChannelId && v.channelId !== state.selectedChannelId) {
        return false;
      }

      // 3. Topic filter
      if (state.selectedTopic !== "all") {
        if (!v.topics || !v.topics.includes(state.selectedTopic)) {
          return false;
        }
      }

      // 4. Kind filter
      if (state.selectedKind !== "all") {
        if (v.kind !== state.selectedKind) {
          return false;
        }
      }

      // 5. Duration bucket filter
      if (!matchesDurationBucket(v.durationSec, state.selectedDuration)) {
        return false;
      }

      // 6. Recency filter
      if (!matchesRecencyBucket(v.publishedAt, state.selectedRecency)) {
        return false;
      }

      // 7. In-memory search query
      if (tokens.length > 0) {
        const score = scoreVideo(v, tokens);
        if (score === 0) return false;
        v._searchScore = score;
      } else {
        v._searchScore = 0;
      }

      return true;
    });

    // Sort
    if (state.selectedSort === "relevant" && tokens.length > 0) {
      list.sort((a, b) => b._searchScore - a._searchScore);
    } else if (state.selectedSort === "newest" || state.selectedSort === "relevant") {
      list.sort((a, b) => (b.publishedAt || "").localeCompare(a.publishedAt || ""));
    } else if (state.selectedSort === "duration-desc") {
      list.sort((a, b) => (b.durationSec || 0) - (a.durationSec || 0));
    } else if (state.selectedSort === "duration-asc") {
      list.sort((a, b) => (a.durationSec || 0) - (b.durationSec || 0));
    } else if (state.selectedSort === "title-asc") {
      list.sort((a, b) => (a.title || "").localeCompare(b.title || ""));
    }

    return list;
  }

  /**
   * Create Video Card Element
   * Click opens YouTube in a new tab.
   * Clicking channel name filters by that channel.
   * Clicking a topic tag filters by that topic.
   */
  function createVideoCardElement(video) {
    const card = document.createElement("article");
    card.className = "video-card";
    card.setAttribute("role", "article");
    card.setAttribute("tabindex", "0");
    card.setAttribute("aria-label", `${video.title} by ${video.channelName}. Opens YouTube in new tab.`);

    const durText = formatDuration(video.durationSec);
    const dateText = formatDate(video.publishedAt);
    const kindLabel = (video.kind || "visual-explainer").replace(/-/g, " ");
    const ytUrl = video.url || `https://www.youtube.com/watch?v=${video.id}`;
    const fallbackThumb = `https://i.ytimg.com/vi/${video.id}/hqdefault.jpg`;
    const thumbUrl = video.thumbnail || fallbackThumb;

    const topicsHtml = (video.topics || [])
      .map((t) => `<button type="button" class="card-topic-tag" data-topic="${escapeHTML(t)}">${escapeHTML(t)}</button>`)
      .join("");

    card.innerHTML = `
      <div class="card-media">
        <img 
          class="card-thumbnail" 
          src="${escapeHTML(thumbUrl)}" 
          alt="${escapeHTML(video.title)}" 
          loading="lazy"
          onerror="this.src='${fallbackThumb}'"
        />
        <span class="card-duration-badge">${escapeHTML(durText)}</span>
        <span class="card-kind-badge">${escapeHTML(kindLabel)}</span>
      </div>
      <div class="card-content">
        <div class="card-channel-row">
          <button type="button" class="card-channel-btn" data-channel-id="${escapeHTML(video.channelId)}" title="Filter catalog by ${escapeHTML(video.channelName)}">
            ${escapeHTML(video.channelName)}
          </button>
          <span class="card-date">${escapeHTML(dateText)}</span>
        </div>
        <h3 class="card-title" title="${escapeHTML(video.title)}">
          ${escapeHTML(video.title)}
        </h3>
        <p class="card-blurb" title="${escapeHTML(video.blurb || '')}">
          ${escapeHTML(video.blurb || "Curated lecture-grade exposition.")}
        </p>
        <div class="card-footer">
          <div class="card-topics">
            ${topicsHtml}
          </div>
          <a 
            href="${escapeHTML(ytUrl)}" 
            target="_blank" 
            rel="noopener noreferrer" 
            class="card-action-link"
            title="Open on YouTube in new tab"
          >
            Watch ↗
          </a>
        </div>
      </div>
    `;

    // Filter by channel when channel button clicked
    const channelBtn = card.querySelector(".card-channel-btn");
    channelBtn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      setChannelFilter(video.channelId);
    });

    // Filter by topic when topic badge clicked
    card.querySelectorAll(".card-topic-tag").forEach((tBtn) => {
      tBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        setTopicFilter(tBtn.dataset.topic);
      });
    });

    // Clicking anywhere on the card opens YouTube in a new tab
    card.addEventListener("click", (e) => {
      if (e.target.closest(".card-channel-btn") || e.target.closest(".card-topic-tag")) {
        return;
      }
      window.open(ytUrl, "_blank", "noopener,noreferrer");
    });

    // Keyboard accessibility (Enter/Space opens YouTube in new tab)
    card.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        if (e.target === card) {
          e.preventDefault();
          window.open(ytUrl, "_blank", "noopener,noreferrer");
        }
      }
    });

    return card;
  }

  /**
   * Render Channel Banner (Show channel blurb and details)
   */
  function renderChannelBanner() {
    if (!state.selectedChannelId) {
      channelFilterBanner.hidden = true;
      return;
    }

    const channel = state.channelsMap.get(state.selectedChannelId);
    const channelVideos = state.videos.filter((v) => v.channelId === state.selectedChannelId);
    const channelName = channel ? channel.name : (channelVideos[0] ? channelVideos[0].channelName : "Faculty");
    const channelKind = channel ? (channel.kind || "curated-channel").replace(/-/g, " ") : "Faculty";
    const channelBlurb = channel ? channel.blurb : (channelVideos[0] ? channelVideos[0].blurb : "Selected faculty in the Lectern canon.");
    const topics = channel && channel.topics ? channel.topics : (channelVideos[0] ? channelVideos[0].topics : []);
    const ytUrl = `https://www.youtube.com/channel/${state.selectedChannelId}`;

    bannerChannelKind.textContent = channelKind;
    bannerChannelName.textContent = channelName;
    bannerChannelBlurb.textContent = channelBlurb;
    bannerChannelCount.textContent = `${channelVideos.length} lecture-grade work${channelVideos.length === 1 ? "" : "s"} in canon`;
    bannerChannelYt.href = ytUrl;

    bannerChannelTopics.innerHTML = (topics || [])
      .map((t) => `<span class="applied-badge">${escapeHTML(t)}</span>`)
      .join("");

    channelFilterBanner.hidden = false;
  }

  /**
   * Render Active Query / Filter Indicator Bar
   */
  function renderActiveQueryBar(filteredCount) {
    const tags = [];

    if (state.searchQuery.trim()) {
      tags.push(`Query: "${state.searchQuery.trim()}"`);
    }
    if (state.selectedTopic !== "all") {
      tags.push(`Topic: ${state.selectedTopic}`);
    }
    if (state.selectedKind !== "all") {
      tags.push(`Kind: ${state.selectedKind.replace(/-/g, " ")}`);
    }
    if (state.selectedDuration !== "all") {
      tags.push(`Duration: ${state.selectedDuration}m`);
    }
    if (state.selectedRecency !== "all") {
      const recLabels = { "past-1": "< 1 yr", "past-3": "< 3 yrs", classic: "Archival" };
      tags.push(`Recency: ${recLabels[state.selectedRecency] || state.selectedRecency}`);
    }
    if (state.selectedChannelId) {
      const ch = state.channelsMap.get(state.selectedChannelId);
      tags.push(`Channel: ${ch ? ch.name : state.selectedChannelId}`);
    }
    if (!state.hideShortsAndUnder12m) {
      tags.push(`Including <12m & Shorts`);
    }

    const hasFilters =
      tags.length > 0 ||
      state.searchQuery.trim().length > 0 ||
      state.selectedSort !== "relevant";

    if (hasFilters) {
      activeQueryBar.hidden = false;
      resultsCountText.textContent = `Showing ${filteredCount.toLocaleString()} of ${state.videos.length.toLocaleString()} works`;
      appliedTags.innerHTML = tags
        .map((tag) => `<span class="applied-badge">${escapeHTML(tag)}</span>`)
        .join("");
    } else {
      activeQueryBar.hidden = true;
    }
  }

  /**
   * Render Main Catalog
   */
  function renderCatalog() {
    renderChannelBanner();

    const filtered = getFilteredVideos();
    videoGrid.innerHTML = "";

    if (filtered.length === 0) {
      videoGrid.hidden = true;
      emptyState.hidden = false;
    } else {
      videoGrid.hidden = false;
      emptyState.hidden = true;

      const frag = document.createDocumentFragment();
      for (const v of filtered) {
        frag.appendChild(createVideoCardElement(v));
      }
      videoGrid.appendChild(frag);
    }

    renderActiveQueryBar(filtered.length);
  }

  /**
   * Render Faculty Grid
   */
  function renderFaculty() {
    facultyGrid.innerHTML = "";
    const list = state.channels;

    if (list.length === 0) {
      facultyGrid.innerHTML = `<p style="grid-column: 1/-1; font-style: italic; color: var(--ink-muted);">No faculty channels currently registered.</p>`;
      return;
    }

    const frag = document.createDocumentFragment();
    for (const ch of list) {
      const count = state.videos.filter((v) => v.channelId === ch.id).length;
      const card = document.createElement("article");
      card.className = "faculty-card";

      const topicsHtml = (ch.topics || [])
        .map((t) => `<span class="card-topic-tag">${escapeHTML(t)}</span>`)
        .join("");

      card.innerHTML = `
        <div class="faculty-card-header">
          <h3 class="faculty-name">${escapeHTML(ch.name)}</h3>
          <span class="faculty-kind">${escapeHTML((ch.kind || "Explainer").replace(/-/g, " "))}</span>
        </div>
        <div class="card-topics" style="margin-bottom: 0.75rem;">
          ${topicsHtml}
        </div>
        <p class="faculty-blurb">${escapeHTML(ch.blurb || "")}</p>
        <div class="faculty-footer">
          <span style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--ink-muted);">
            ${count} curated work${count === 1 ? "" : "s"}
          </span>
          <div class="faculty-actions">
            <a href="https://www.youtube.com/channel/${escapeHTML(ch.id)}" target="_blank" rel="noopener noreferrer" class="faculty-yt-link">
              YouTube ↗
            </a>
            <button type="button" class="faculty-filter-btn" data-channel-id="${escapeHTML(ch.id)}">
              View Lectures
            </button>
          </div>
        </div>
      `;

      const filterBtn = card.querySelector(".faculty-filter-btn");
      filterBtn.addEventListener("click", () => {
        setChannelFilter(ch.id);
      });

      frag.appendChild(card);
    }
    facultyGrid.appendChild(frag);
  }

  /**
   * Render Courses / Playlists Grid
   */
  function renderCourses() {
    coursesGrid.innerHTML = "";
    const list = state.playlists;

    if (list.length === 0) {
      coursesGrid.innerHTML = `<p style="grid-column: 1/-1; font-style: italic; color: var(--ink-muted);">No course playlists registered or playlists catalog unavailable.</p>`;
      return;
    }

    const frag = document.createDocumentFragment();
    for (const pl of list) {
      const card = document.createElement("article");
      card.className = "course-card";

      const topicsHtml = (pl.topics || [])
        .map((t) => `<span class="card-topic-tag">${escapeHTML(t)}</span>`)
        .join("");

      const ytUrl = `https://www.youtube.com/playlist?list=${pl.id}`;

      card.innerHTML = `
        <span class="course-channel">${escapeHTML(pl.channelName)}</span>
        <h3 class="course-title">${escapeHTML(pl.title)}</h3>
        <p class="course-blurb">${escapeHTML(pl.blurb || "")}</p>
        <div class="course-footer">
          <div class="card-topics">
            ${topicsHtml}
          </div>
          <a href="${escapeHTML(ytUrl)}" target="_blank" rel="noopener noreferrer" class="course-action-link">
            Open Course on YouTube ↗
          </a>
        </div>
      `;

      frag.appendChild(card);
    }
    coursesGrid.appendChild(frag);
  }

  /**
   * View Mode Switcher
   */
  function setViewMode(mode) {
    state.currentView = mode;

    viewModeCatalog.classList.toggle("active", mode === "catalog");
    viewModeCatalog.setAttribute("aria-selected", mode === "catalog" ? "true" : "false");
    viewModeChannels.classList.toggle("active", mode === "channels");
    viewModeChannels.setAttribute("aria-selected", mode === "channels" ? "true" : "false");
    viewModePlaylists.classList.toggle("active", mode === "playlists");
    viewModePlaylists.setAttribute("aria-selected", mode === "playlists" ? "true" : "false");

    catalogSection.hidden = mode !== "catalog";
    channelsSection.hidden = mode !== "channels";
    coursesSection.hidden = mode !== "playlists";

    if (mode === "catalog") {
      renderCatalog();
    } else if (mode === "channels") {
      renderFaculty();
    } else if (mode === "playlists") {
      renderCourses();
    }
  }

  /**
   * Channel Filter Trigger
   */
  function setChannelFilter(channelId) {
    state.selectedChannelId = channelId;
    setViewMode("catalog");
    window.scrollTo({ top: 0, behavior: "smooth" });
    renderCatalog();
  }

  /**
   * Topic Filter Trigger
   */
  function setTopicFilter(topic) {
    state.selectedTopic = topic;
    topicChips.querySelectorAll(".chip").forEach((chip) => {
      chip.classList.toggle("active", chip.dataset.topic === topic);
    });
    setViewMode("catalog");
    renderCatalog();
  }

  /**
   * Reset All Filters to Canon Defaults
   */
  function resetAllFilters() {
    state.searchQuery = "";
    state.selectedTopic = "all";
    state.selectedKind = "all";
    state.selectedDuration = "all";
    state.selectedRecency = "all";
    state.selectedSort = "relevant";
    state.selectedChannelId = null;
    state.hideShortsAndUnder12m = true;

    searchInput.value = "";
    clearSearchBtn.hidden = true;

    // Reset Chip Visuals
    document.querySelectorAll(".chip").forEach((chip) => {
      const topic = chip.dataset.topic;
      const kind = chip.dataset.kind;
      const duration = chip.dataset.duration;
      const recency = chip.dataset.recency;
      const sort = chip.dataset.sort;

      if (topic) chip.classList.toggle("active", topic === "all");
      if (kind) chip.classList.toggle("active", kind === "all");
      if (duration) chip.classList.toggle("active", duration === "all");
      if (recency) chip.classList.toggle("active", recency === "all");
      if (sort) chip.classList.toggle("active", sort === "relevant");
    });

    toggleShortsGuardBtn.classList.add("active");
    toggleShortsGuardBtn.querySelector(".guard-indicator").textContent = "✓";

    setViewMode("catalog");
    renderCatalog();
  }

  /**
   * Event Listeners Setup
   */
  function setupEvents() {
    // Search input
    searchInput.addEventListener("input", (e) => {
      state.searchQuery = e.target.value;
      clearSearchBtn.hidden = state.searchQuery.length === 0;
      if (state.currentView !== "catalog") {
        setViewMode("catalog");
      } else {
        renderCatalog();
      }
    });

    clearSearchBtn.addEventListener("click", () => {
      searchInput.value = "";
      state.searchQuery = "";
      clearSearchBtn.hidden = true;
      renderCatalog();
      searchInput.focus();
    });

    // Topic chips
    topicChips.addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-topic]");
      if (!btn) return;
      topicChips.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
      btn.classList.add("active");
      state.selectedTopic = btn.dataset.topic;
      if (state.currentView !== "catalog") setViewMode("catalog");
      renderCatalog();
    });

    // Kind chips
    kindChips.addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-kind]");
      if (!btn) return;
      kindChips.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
      btn.classList.add("active");
      state.selectedKind = btn.dataset.kind;
      if (state.currentView !== "catalog") setViewMode("catalog");
      renderCatalog();
    });

    // Duration chips
    durationChips.addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-duration]");
      if (!btn) return;
      durationChips.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
      btn.classList.add("active");
      state.selectedDuration = btn.dataset.duration;
      if (state.currentView !== "catalog") setViewMode("catalog");
      renderCatalog();
    });

    // Recency chips
    recencyChips.addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-recency]");
      if (!btn) return;
      recencyChips.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
      btn.classList.add("active");
      state.selectedRecency = btn.dataset.recency;
      if (state.currentView !== "catalog") setViewMode("catalog");
      renderCatalog();
    });

    // Sort chips
    sortChips.addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-sort]");
      if (!btn) return;
      sortChips.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
      btn.classList.add("active");
      state.selectedSort = btn.dataset.sort;
      renderCatalog();
    });

    // Toggle Shorts & <12m Guard
    toggleShortsGuardBtn.addEventListener("click", () => {
      state.hideShortsAndUnder12m = !state.hideShortsAndUnder12m;
      toggleShortsGuardBtn.classList.toggle("active", state.hideShortsAndUnder12m);
      toggleShortsGuardBtn.querySelector(".guard-indicator").textContent = state.hideShortsAndUnder12m ? "✓" : "○";
      renderCatalog();
    });

    // Channel Banner Clear Button
    bannerChannelClear.addEventListener("click", () => {
      state.selectedChannelId = null;
      renderCatalog();
    });

    // Reset filters
    resetFiltersBtn.addEventListener("click", resetAllFilters);
    emptyResetBtn.addEventListener("click", resetAllFilters);

    // View toggles
    viewModeCatalog.addEventListener("click", () => setViewMode("catalog"));
    viewModeChannels.addEventListener("click", () => setViewMode("channels"));
    viewModePlaylists.addEventListener("click", () => setViewMode("playlists"));

    // Global keyboard shortcut: Escape clears query or channel focus
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        if (state.searchQuery || state.selectedChannelId) {
          if (state.searchQuery) {
            state.searchQuery = "";
            searchInput.value = "";
            clearSearchBtn.hidden = true;
          } else if (state.selectedChannelId) {
            state.selectedChannelId = null;
          }
          renderCatalog();
        }
      }
    });
  }

  /**
   * Load JSON Datasets defensively
   */
  async function loadData() {
    try {
      const [vData, cData, pData] = await Promise.all([
        fetchJsonDefensive("videos.json"),
        fetchJsonDefensive("channels.json"),
        fetchJsonDefensive("playlists.json"),
      ]);

      state.videos = extractVideos(vData);
      state.channels = extractChannels(cData);
      state.playlists = extractPlaylists(pData);

      // If channels.json was missing or empty, build channels defensive map from videos
      if (state.channels.length === 0 && state.videos.length > 0) {
        const derivedMap = new Map();
        for (const v of state.videos) {
          if (v.channelId && !derivedMap.has(v.channelId)) {
            derivedMap.set(v.channelId, {
              id: v.channelId,
              name: v.channelName || "Faculty",
              topics: v.topics || [],
              kind: v.kind || "visual-explainer",
              blurb: v.blurb || "Lecture-grade creator in the Lectern canon.",
            });
          }
        }
        state.channels = Array.from(derivedMap.values());
      }

      // Populate channels fast lookup map
      state.channelsMap.clear();
      for (const ch of state.channels) {
        state.channelsMap.set(ch.id, ch);
      }

      // Update counters
      statVideos.textContent = state.videos.length.toLocaleString();
      statChannels.textContent = state.channels.length.toLocaleString();

      // If playlists missing, disable playlists button gracefully
      if (state.playlists.length === 0) {
        viewModePlaylists.title = "No course playlists available";
      }

      renderCatalog();
    } catch (err) {
      console.error("Failed to load Lectern data:", err);
      videoGrid.innerHTML = `
        <div class="empty-state" style="grid-column: 1 / -1;">
          <h2 class="empty-title">Initialization Error</h2>
          <p class="empty-copy">Unable to load library data catalogs. Please run a local web server (e.g. <code>python3 -m http.server</code>) from the repository root.</p>
        </div>
      `;
    }
  }

  // Initialize Application
  setupEvents();
  loadData();
})();

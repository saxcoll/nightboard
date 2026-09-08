/**
 * Lectern — Academic Library Client Application
 * Handles in-memory search, multi-faceted filtering (topic, kind, duration, channel),
 * view mode switching, and accessible modal views.
 */

(function () {
  "use strict";

  // Application State
  const state = {
    videos: [],
    channels: [],
    playlists: [],
    searchQuery: "",
    selectedTopic: "all",
    selectedKind: "all",
    selectedDuration: "all",
    selectedChannelId: null,
    selectedSort: "relevant",
    currentView: "catalog", // 'catalog' | 'channels' | 'playlists'
  };

  // DOM Elements
  const searchInput = document.getElementById("library-search");
  const clearSearchBtn = document.getElementById("clear-search-btn");
  const topicChips = document.getElementById("topic-chips");
  const kindChips = document.getElementById("kind-chips");
  const durationChips = document.getElementById("duration-chips");
  const sortChips = document.getElementById("sort-chips");
  const activeQueryBar = document.getElementById("active-query-bar");
  const resultsCountText = document.getElementById("results-count-text");
  const appliedTags = document.getElementById("applied-tags");
  const resetFiltersBtn = document.getElementById("reset-filters-btn");
  const emptyResetBtn = document.getElementById("empty-reset-btn");

  const videoGrid = document.getElementById("video-grid");
  const facultyGrid = document.getElementById("faculty-grid");
  const coursesGrid = document.getElementById("courses-grid");
  const emptyState = document.getElementById("empty-state");

  const catalogSection = document.getElementById("catalog-section");
  const channelsSection = document.getElementById("channels-section");
  const coursesSection = document.getElementById("courses-section");

  const viewModeCatalog = document.getElementById("view-mode-catalog");
  const viewModeChannels = document.getElementById("view-mode-channels");
  const viewModePlaylists = document.getElementById("view-mode-playlists");

  const statChannels = document.getElementById("stat-channels");
  const statVideos = document.getElementById("stat-videos");

  // Modals
  const channelModal = document.getElementById("channel-modal");
  const modalChannelName = document.getElementById("modal-channel-name");
  const modalChannelKind = document.getElementById("modal-channel-kind");
  const modalChannelBlurb = document.getElementById("modal-channel-blurb");
  const modalChannelTopics = document.getElementById("modal-channel-topics");
  const modalChannelYtLink = document.getElementById("modal-channel-yt-link");
  const modalChannelCount = document.getElementById("modal-channel-count");
  const modalChannelVideos = document.getElementById("modal-channel-videos");
  const modalCloseBtn = document.getElementById("modal-close-btn");

  const videoModal = document.getElementById("video-modal");
  const modalVideoChannel = document.getElementById("modal-video-channel");
  const modalVideoTitle = document.getElementById("modal-video-title");
  const videoPlayerContainer = document.getElementById("video-player-container");
  const playerDurationBadge = document.getElementById("player-duration-badge");
  const playerKindBadge = document.getElementById("player-kind-badge");
  const playerDirectYtLink = document.getElementById("player-direct-yt-link");
  const videoModalCloseBtn = document.getElementById("video-modal-close-btn");

  /**
   * Format duration in seconds to "MM:SS" or "H:MM:SS"
   */
  function formatDuration(seconds) {
    if (seconds == null || isNaN(seconds)) return "—";
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) {
      return `${h}:${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
    }
    return `${m}:${s.toString().padStart(2, "0")}`;
  }

  /**
   * Format ISO date string into readable academic format
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
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  /**
   * Match duration buckets
   */
  function matchesDurationBucket(sec, bucket) {
    if (bucket === "all") return true;
    if (sec == null) return false; // exclude unknown when filtering specific duration
    const min = sec / 60;
    if (bucket === "12-20") return min >= 12 && min < 20;
    if (bucket === "20-45") return min >= 20 && min < 45;
    if (bucket === "45-90") return min >= 45 && min < 90;
    if (bucket === "90+") return min >= 90;
    return true;
  }

  /**
   * Token-based search ranking score
   */
  function scoreVideo(video, queryTokens) {
    if (queryTokens.length === 0) return 1;

    const titleLower = (video.title || "").toLowerCase();
    const channelLower = (video.channelName || "").toLowerCase();
    const blurbLower = (video.blurb || "").toLowerCase();
    const topicsLower = (video.topics || []).join(" ").toLowerCase();

    let score = 0;
    for (const token of queryTokens) {
      if (titleLower.includes(token)) score += 10;
      else if (channelLower.includes(token)) score += 6;
      else if (topicsLower.includes(token)) score += 4;
      else if (blurbLower.includes(token)) score += 2;
      else return 0; // All tokens must match somewhere (AND condition)
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
      // 1. Minimum duration filter (enforced >= 12 min when duration is known)
      if (v.durationSec != null && v.durationSec < 720) return false;

      // 2. Channel filter (if channel view activated)
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

      // 5. Duration bucket
      if (!matchesDurationBucket(v.durationSec, state.selectedDuration)) {
        return false;
      }

      // 6. Search query
      if (tokens.length > 0) {
        const score = scoreVideo(v, tokens);
        if (score === 0) return false;
        v._searchScore = score;
      } else {
        v._searchScore = 0;
      }

      return true;
    });

    // Sort list
    if (state.selectedSort === "relevant" && tokens.length > 0) {
      list.sort((a, b) => b._searchScore - a._searchScore);
    } else if (state.selectedSort === "duration-desc") {
      list.sort((a, b) => (b.durationSec || 0) - (a.durationSec || 0));
    } else if (state.selectedSort === "duration-asc") {
      list.sort((a, b) => (a.durationSec || 0) - (b.durationSec || 0));
    } else if (state.selectedSort === "title-asc") {
      list.sort((a, b) => (a.title || "").localeCompare(b.title || ""));
    } else {
      // Default: sort by date or catalog order
      list.sort((a, b) => (b.publishedAt || "").localeCompare(a.publishedAt || ""));
    }

    return list;
  }

  /**
   * Render video card HTML
   */
  function createVideoCardElement(video) {
    const card = document.createElement("article");
    card.className = "video-card";
    card.setAttribute("role", "article");
    card.setAttribute("tabindex", "0");

    const durText = formatDuration(video.durationSec);
    const dateText = formatDate(video.publishedAt);
    const kindLabel = (video.kind || "visual-explainer").replace("-", " ");
    const topicsHtml = (video.topics || [])
      .map((t) => `<span class="card-topic-tag">${escapeHTML(t)}</span>`)
      .join("");

    card.innerHTML = `
      <div class="card-media">
        <img 
          class="card-thumbnail" 
          src="${escapeHTML(video.thumbnail)}" 
          alt="${escapeHTML(video.title)}" 
          loading="lazy"
          onerror="this.src='https://i.ytimg.com/vi/${video.id}/hqdefault.jpg'"
        />
        <span class="card-duration-badge">${escapeHTML(durText)}</span>
        <span class="card-kind-badge">${escapeHTML(kindLabel)}</span>
      </div>
      <div class="card-content">
        <div class="card-channel-row">
          <a href="#" class="card-channel-name" data-channel-id="${escapeHTML(video.channelId)}">
            ${escapeHTML(video.channelName)}
          </a>
          <span class="card-date">${escapeHTML(dateText)}</span>
        </div>
        <h3 class="card-title" title="${escapeHTML(video.title)}">
          ${escapeHTML(video.title)}
        </h3>
        <p class="card-blurb" title="${escapeHTML(video.blurb || '')}">
          ${escapeHTML(video.blurb || "Lecture-grade exposition from the curated canon.")}
        </p>
        <div class="card-footer">
          <div class="card-topics">
            ${topicsHtml}
          </div>
          <a 
            href="${escapeHTML(video.url)}" 
            target="_blank" 
            rel="noopener noreferrer" 
            class="card-action-link"
            title="Open on YouTube"
          >
            Watch ↗
          </a>
        </div>
      </div>
    `;

    // Click on channel name filters to that channel or opens modal
    const channelLink = card.querySelector(".card-channel-name");
    channelLink.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      openChannelModal(video.channelId);
    });

    // Clicking the card opens the video player modal (with direct link)
    card.addEventListener("click", (e) => {
      // If clicked on watch link directly, let default action happen
      if (e.target.closest(".card-action-link")) return;
      openVideoModal(video);
    });

    card.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        openVideoModal(video);
      }
    });

    return card;
  }

  /**
   * Render catalog
   */
  function renderCatalog() {
    const filtered = getFilteredVideos();
    videoGrid.innerHTML = "";

    if (filtered.length === 0) {
      videoGrid.hidden = true;
      emptyState.hidden = false;
    } else {
      videoGrid.hidden = false;
      emptyState.hidden = false; // toggle below
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
   * Render Active Query / Filter Indicator Bar
   */
  function renderActiveQueryBar(count) {
    const tags = [];

    if (state.searchQuery.trim()) {
      tags.push(`Query: "${state.searchQuery.trim()}"`);
    }
    if (state.selectedTopic !== "all") {
      tags.push(`Topic: ${state.selectedTopic}`);
    }
    if (state.selectedKind !== "all") {
      tags.push(`Kind: ${state.selectedKind}`);
    }
    if (state.selectedDuration !== "all") {
      tags.push(`Duration: ${state.selectedDuration}m`);
    }
    if (state.selectedChannelId) {
      const ch = state.channels.find((c) => c.id === state.selectedChannelId);
      tags.push(`Faculty: ${ch ? ch.name : state.selectedChannelId}`);
    }

    if (tags.length > 0) {
      activeQueryBar.hidden = false;
      resultsCountText.textContent = `Showing ${count} works in selection`;
      appliedTags.innerHTML = tags
        .map((t) => `<span class="applied-badge">${escapeHTML(t)}</span>`)
        .join("");
    } else {
      activeQueryBar.hidden = false;
      resultsCountText.textContent = `Showing all ${count} works in library`;
      appliedTags.innerHTML = `<span class="applied-badge">Full Library</span>`;
    }
  }

  /**
   * Render Faculty / Channels View
   */
  function renderFaculty() {
    facultyGrid.innerHTML = "";
    const frag = document.createDocumentFragment();

    for (const ch of state.channels) {
      const card = document.createElement("article");
      card.className = "faculty-card";

      const topicsHtml = (ch.topics || [])
        .map((t) => `<span class="card-topic-tag">${escapeHTML(t)}</span>`)
        .join("");

      card.innerHTML = `
        <div class="faculty-header">
          <div>
            <h3 class="faculty-title">${escapeHTML(ch.name)}</h3>
            <span class="faculty-kind-tag">${escapeHTML(ch.kind)}</span>
          </div>
        </div>
        <p class="faculty-blurb">${escapeHTML(ch.blurb)}</p>
        <div class="faculty-footer">
          <div class="card-topics">${topicsHtml}</div>
          <button type="button" class="faculty-filter-btn" data-channel-id="${escapeHTML(ch.id)}">
            View Works ↗
          </button>
        </div>
      `;

      card.querySelector(".faculty-filter-btn").addEventListener("click", () => {
        openChannelModal(ch.id);
      });

      frag.appendChild(card);
    }
    facultyGrid.appendChild(frag);
  }

  /**
   * Render Courses / Playlists View
   */
  function renderCourses() {
    coursesGrid.innerHTML = "";
    const frag = document.createDocumentFragment();

    for (const pl of state.playlists) {
      const card = document.createElement("article");
      card.className = "course-card";

      const topicsHtml = (pl.topics || [])
        .map((t) => `<span class="card-topic-tag">${escapeHTML(t)}</span>`)
        .join("");

      card.innerHTML = `
        <h3>${escapeHTML(pl.title)}</h3>
        <p class="course-channel">${escapeHTML(pl.channelName)} · ${escapeHTML(pl.kind)}</p>
        <p class="course-blurb">${escapeHTML(pl.blurb)}</p>
        <div class="course-footer">
          <div class="card-topics">${topicsHtml}</div>
          <a 
            href="https://www.youtube.com/playlist?list=${escapeHTML(pl.id)}" 
            target="_blank" 
            rel="noopener noreferrer" 
            class="course-yt-link"
          >
            Open Playlist on YouTube ↗
          </a>
        </div>
      `;
      frag.appendChild(card);
    }
    coursesGrid.appendChild(frag);
  }

  /**
   * Switch View Mode
   */
  function setViewMode(mode) {
    state.currentView = mode;
    viewModeCatalog.classList.toggle("active", mode === "catalog");
    viewModeChannels.classList.toggle("active", mode === "channels");
    viewModePlaylists.classList.toggle("active", mode === "playlists");

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
   * Channel Modal
   */
  function openChannelModal(channelId) {
    const ch = state.channels.find((c) => c.id === channelId);
    if (!ch) return;

    modalChannelName.textContent = ch.name;
    modalChannelKind.textContent = (ch.kind || "Visual Explainer").replace("-", " ");
    modalChannelBlurb.textContent = ch.blurb;
    modalChannelYtLink.href = `https://www.youtube.com/channel/${ch.id}`;

    modalChannelTopics.innerHTML = (ch.topics || [])
      .map((t) => `<span class="card-topic-tag">${escapeHTML(t)}</span>`)
      .join("");

    const channelVids = state.videos.filter((v) => v.channelId === ch.id);
    modalChannelCount.textContent = channelVids.length;

    modalChannelVideos.innerHTML = "";
    if (channelVids.length === 0) {
      modalChannelVideos.innerHTML = `<p style="font-style:italic;color:var(--ink-muted);">No videos currently ingested for this faculty member.</p>`;
    } else {
      const frag = document.createDocumentFragment();
      for (const v of channelVids) {
        const item = document.createElement("div");
        item.className = "video-card";
        item.innerHTML = `
          <div class="card-media">
            <img class="card-thumbnail" src="${escapeHTML(v.thumbnail)}" alt="" loading="lazy"/>
            <span class="card-duration-badge">${escapeHTML(formatDuration(v.durationSec))}</span>
          </div>
          <div class="card-content" style="padding:0.75rem;">
            <h4 style="font-size:0.95rem;font-weight:700;line-height:1.3;margin-bottom:0.4rem;">${escapeHTML(v.title)}</h4>
            <a href="${escapeHTML(v.url)}" target="_blank" rel="noopener noreferrer" class="card-action-link" style="font-size:0.75rem;">Watch ↗</a>
          </div>
        `;
        frag.appendChild(item);
      }
      modalChannelVideos.appendChild(frag);
    }

    channelModal.hidden = false;
    document.body.style.overflow = "hidden";
  }

  function closeChannelModal() {
    channelModal.hidden = true;
    document.body.style.overflow = "";
  }

  /**
   * Video Modal
   */
  function openVideoModal(video) {
    modalVideoChannel.textContent = video.channelName;
    modalVideoTitle.textContent = video.title;
    playerDurationBadge.textContent = formatDuration(video.durationSec);
    playerKindBadge.textContent = (video.kind || "Visual Explainer").replace("-", " ");
    playerDirectYtLink.href = video.url;

    // Modest YouTube embed
    videoPlayerContainer.innerHTML = `
      <iframe 
        src="https://www.youtube-nocookie.com/embed/${video.id}?autoplay=1&modestbranding=1&rel=0" 
        title="${escapeHTML(video.title)}" 
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
        allowfullscreen
      ></iframe>
    `;

    videoModal.hidden = false;
    document.body.style.overflow = "hidden";
  }

  function closeVideoModal() {
    videoModal.hidden = true;
    videoPlayerContainer.innerHTML = "";
    document.body.style.overflow = "";
  }

  /**
   * Reset All Filters
   */
  function resetAllFilters() {
    state.searchQuery = "";
    state.selectedTopic = "all";
    state.selectedKind = "all";
    state.selectedDuration = "all";
    state.selectedChannelId = null;
    state.selectedSort = "relevant";

    searchInput.value = "";
    clearSearchBtn.hidden = true;

    // Reset chip active states
    document.querySelectorAll(".chip").forEach((chip) => {
      const topic = chip.dataset.topic;
      const kind = chip.dataset.kind;
      const duration = chip.dataset.duration;
      const sort = chip.dataset.sort;

      if (topic) chip.classList.toggle("active", topic === "all");
      if (kind) chip.classList.toggle("active", kind === "all");
      if (duration) chip.classList.toggle("active", duration === "all");
      if (sort) chip.classList.toggle("active", sort === "relevant");
    });

    setViewMode("catalog");
    renderCatalog();
  }

  /**
   * Setup Event Listeners
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

    // Sort chips
    sortChips.addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-sort]");
      if (!btn) return;
      sortChips.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
      btn.classList.add("active");
      state.selectedSort = btn.dataset.sort;
      renderCatalog();
    });

    // Reset filters
    resetFiltersBtn.addEventListener("click", resetAllFilters);
    emptyResetBtn.addEventListener("click", resetAllFilters);

    // View toggles
    viewModeCatalog.addEventListener("click", () => setViewMode("catalog"));
    viewModeChannels.addEventListener("click", () => setViewMode("channels"));
    viewModePlaylists.addEventListener("click", () => setViewMode("playlists"));

    // Modal close events
    modalCloseBtn.addEventListener("click", closeChannelModal);
    channelModal.addEventListener("click", (e) => {
      if (e.target === channelModal) closeChannelModal();
    });

    videoModalCloseBtn.addEventListener("click", closeVideoModal);
    videoModal.addEventListener("click", (e) => {
      if (e.target === videoModal) closeVideoModal();
    });

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        if (!channelModal.hidden) closeChannelModal();
        if (!videoModal.hidden) closeVideoModal();
      }
    });
  }

  /**
   * Load JSON Datasets
   */
  async function loadData() {
    try {
      const [vRes, cRes, pRes] = await Promise.all([
        fetch("./data/videos.json"),
        fetch("./data/channels.json"),
        fetch("./data/playlists.json"),
      ]);

      if (vRes.ok) state.videos = await vRes.json();
      if (cRes.ok) state.channels = await cRes.json();
      if (pRes.ok) state.playlists = await pRes.json();

      statChannels.textContent = state.channels.length || "62";
      statVideos.textContent = state.videos.length || "0";

      renderCatalog();
    } catch (err) {
      console.error("Failed to load Lectern data:", err);
      videoGrid.innerHTML = `
        <div class="empty-state" style="grid-column: 1 / -1;">
          <h2 class="empty-title">Initialization Error</h2>
          <p class="empty-copy">Unable to load library data catalogs. Please verify python server is active.</p>
        </div>
      `;
    }
  }

  // Initialize
  setupEvents();
  loadData();
})();

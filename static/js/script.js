/**
 * Darts Throw Analysis AI — Phase 1 frontend
 *
 * Responsible for:
 *   - Uploading videos (drag & drop or file picker) to /api/upload
 *   - Rendering the video library and wiring select / rename / delete
 *   - Driving the custom video player: play/pause/stop, frame stepping,
 *     timeline scrubbing, volume, fullscreen
 *   - The 0.10x-2.00x playback speed slider
 *   - Showing the "coming in Version 2" message for analysis
 *
 * No pose detection or scoring logic lives here yet — the Analysis panel
 * only ever shows placeholders in Phase 1.
 */

(() => {
  "use strict";

  const MAX_VIDEOS = 20;

  /* ------------------------------------------------------------------ *
   * Element references
   * ------------------------------------------------------------------ */

  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const uploadBtn = document.getElementById("upload-btn");
  const addMoreBtn = document.getElementById("add-more-btn");
  const uploadFeedback = document.getElementById("upload-feedback");

  const recordBtn = document.getElementById("record-btn");
  const recordMoreBtn = document.getElementById("record-more-btn");
  const recordModal = document.getElementById("record-modal");
  const recordPreview = document.getElementById("record-preview");
  const recordIndicator = document.getElementById("record-indicator");
  const recordTimerEl = document.getElementById("record-timer");
  const recordHint = document.getElementById("record-hint");
  const recordCancelBtn = document.getElementById("record-cancel");
  const recordStartBtn = document.getElementById("record-start");
  const recordStopBtn = document.getElementById("record-stop");
  const recordRetakeBtn = document.getElementById("record-retake");
  const recordUseBtn = document.getElementById("record-use");

  const youtubeBtn = document.getElementById("youtube-btn");
  const youtubeMoreBtn = document.getElementById("youtube-more-btn");
  const youtubeModal = document.getElementById("youtube-modal");
  const youtubeInput = document.getElementById("youtube-input");
  const youtubeStatus = document.getElementById("youtube-status");
  const youtubeCancelBtn = document.getElementById("youtube-cancel");
  const youtubeConfirmBtn = document.getElementById("youtube-confirm");

  const trimMarkStartBtn = document.getElementById("trim-mark-start");
  const trimStartReadout = document.getElementById("trim-start-readout");
  const trimMarkEndBtn = document.getElementById("trim-mark-end");
  const trimEndReadout = document.getElementById("trim-end-readout");
  const trimCreateBtn = document.getElementById("trim-create-btn");
  const trimStatusEl = document.getElementById("trim-status");

  const hero = document.getElementById("hero");
  const workspace = document.getElementById("workspace");

  const libraryGrid = document.getElementById("library-grid");
  const libraryCount = document.getElementById("library-count");
  const videoCountPill = document.getElementById("video-count-pill");

  const videoEl = document.getElementById("video-player");
  const stageEmpty = document.getElementById("stage-empty");
  const fullscreenBtn = document.getElementById("fullscreen-btn");

  const timelineTrack = document.getElementById("timeline-track");
  const timelineProgress = document.getElementById("timeline-progress");
  const timelineHandle = document.getElementById("timeline-handle");
  const releaseMarker = document.getElementById("release-marker");
  const currentTimeEl = document.getElementById("current-time");
  const totalTimeEl = document.getElementById("total-time");
  const frameReadout = document.getElementById("frame-readout");

  const playBtn = document.getElementById("play-btn");
  const playIcon = document.getElementById("play-icon");
  const stopBtn = document.getElementById("stop-btn");
  const prevFrameBtn = document.getElementById("prev-frame-btn");
  const nextFrameBtn = document.getElementById("next-frame-btn");
  const muteBtn = document.getElementById("mute-btn");
  const volumeSlider = document.getElementById("volume-slider");

  const speedSlider = document.getElementById("speed-slider");
  const speedValue = document.getElementById("speed-value");
  const speedPresets = document.getElementById("speed-presets");

  const armSelect = document.getElementById("arm-select");
  const analyzeBtn = document.getElementById("analyze-btn");
  const analysisStatusPill = document.getElementById("analysis-status-pill");
  const analysisElbow = document.getElementById("analysis-elbow");
  const analysisElbowDirection = document.getElementById("analysis-elbow-direction");
  const analysisWrist = document.getElementById("analysis-wrist");
  const analysisRelease = document.getElementById("analysis-release");
  const analysisOverall = document.getElementById("analysis-overall");
  const analysisNote = document.getElementById("analysis-note");
  const comparisonSummary = document.getElementById("comparison-summary");
  const throwPictograph = document.getElementById("throw-pictograph");

  const coachModeGroup = document.getElementById("coach-mode");
  const coachLog = document.getElementById("coach-log");
  const coachLogEmpty = document.getElementById("coach-log-empty");
  const coachForm = document.getElementById("coach-form");
  const coachInput = document.getElementById("coach-input");
  const coachSend = document.getElementById("coach-send");

  const apiKeyChangeBtn = document.getElementById("api-key-change-btn");
  const apiKeyModal = document.getElementById("api-key-modal");
  const apiKeyProviderSelect = document.getElementById("api-key-provider");
  const apiKeyConsoleLink = document.getElementById("api-key-console-link");
  const apiKeyInput = document.getElementById("api-key-input");
  const apiKeyCancelBtn = document.getElementById("api-key-cancel");
  const apiKeySaveBtn = document.getElementById("api-key-save");
  const apiKeyClearBtn = document.getElementById("api-key-clear");

  const aiAnalysisKeyLink = document.getElementById("ai-analysis-key-link");
  const aiAnalyzeBtn = document.getElementById("ai-analyze-btn");
  const aiAnalysisResult = document.getElementById("ai-analysis-result");

  const infoFilename = document.getElementById("info-filename");
  const infoResolution = document.getElementById("info-resolution");
  const infoFps = document.getElementById("info-fps");
  const infoDuration = document.getElementById("info-duration");
  const infoSize = document.getElementById("info-size");

  const renameModal = document.getElementById("rename-modal");
  const renameInput = document.getElementById("rename-input");
  const renameCancel = document.getElementById("rename-cancel");
  const renameConfirm = document.getElementById("rename-confirm");

  const toastEl = document.getElementById("toast");

  /* ------------------------------------------------------------------ *
   * State
   * ------------------------------------------------------------------ */

  let library = [];          // list of video metadata objects from the server
  let activeVideoId = null;  // id of the video currently loaded in the player
  let renameTargetId = null; // id of the video being renamed via the modal
  let isScrubbing = false;
  let toastTimer = null;
  let selectedThrowIndex = 0;  // which throw's detail is shown, within the active video's analysis.throws
  let coachMode = "rule";      // "rule" | "llm"
  let coachBusy = false;
  let selectedArm = "right";   // "right" | "left" — which arm to analyze
  let trimStart = null;        // seconds — marked in-point for the trim toolbar
  let trimEnd = null;          // seconds — marked out-point for the trim toolbar
  let trimBusy = false;
  let pendingApiKeyCallback = null;   // run once a key is saved, when the modal was opened to unblock an action
  let apiKeyModalFromCoachToggle = false; // whether Cancel should fall back the coach mode toggle to "rule"
  let aiAnalysisBusy = false;

  let mediaStream = null;      // active getUserMedia stream, while the record modal is open
  let mediaRecorder = null;
  let recordedChunks = [];
  let recordedBlob = null;
  let recordTimerInterval = null;
  let recordStartedAt = 0;

  /* ------------------------------------------------------------------ *
   * Helpers
   * ------------------------------------------------------------------ */

  function showToast(message, duration = 2600) {
    toastEl.textContent = message;
    toastEl.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { toastEl.hidden = true; }, duration);
  }

  function setUploadFeedback(message, kind) {
    uploadFeedback.textContent = message || "";
    uploadFeedback.classList.remove("is-error", "is-success");
    if (kind) uploadFeedback.classList.add(kind === "error" ? "is-error" : "is-success");
  }

  function formatTime(seconds) {
    if (!isFinite(seconds) || seconds < 0) seconds = 0;
    const mins = Math.floor(seconds / 60);
    const secs = seconds - mins * 60;
    return `${String(mins).padStart(2, "0")}:${secs.toFixed(2).padStart(5, "0")}`;
  }

  function activeVideo() {
    return library.find((v) => v.id === activeVideoId) || null;
  }

  function frameDuration(video) {
    // Seconds per frame, used for frame-stepping and the frame counter.
    const fps = video && video.fps ? video.fps : 30;
    return 1 / fps;
  }

  /* ------------------------------------------------------------------ *
   * Library rendering
   * ------------------------------------------------------------------ */

  function renderLibrary() {
    libraryGrid.innerHTML = "";

    if (library.length === 0) {
      libraryGrid.innerHTML = `<p class="library__empty">No throws uploaded yet.</p>`;
    }

    library.forEach((video) => {
      const card = document.createElement("div");
      card.className = "video-card" + (video.id === activeVideoId ? " is-active" : "");
      card.dataset.id = video.id;
      card.setAttribute("role", "button");
      card.setAttribute("tabindex", "0");

      card.innerHTML = `
        <img class="video-card__thumb" src="${video.thumbnail_url}" alt="" loading="lazy">
        <div class="video-card__meta">
          <div class="video-card__name">${escapeHtml(video.display_name)}</div>
          <div class="video-card__sub">${formatTime(video.duration)} &middot; ${video.file_size_readable}</div>
        </div>
        <div class="video-card__actions">
          <button class="video-card__icon-btn rename" title="Rename" aria-label="Rename">
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4 12.5-12.5z" stroke-linecap="round" stroke-linejoin="round"/></svg>
          </button>
          <button class="video-card__icon-btn delete" title="Delete" aria-label="Delete">
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13" stroke-linecap="round" stroke-linejoin="round"/></svg>
          </button>
        </div>
      `;

      card.addEventListener("click", (e) => {
        if (e.target.closest(".rename")) { openRenameModal(video.id); return; }
        if (e.target.closest(".delete")) { deleteVideo(video.id); return; }
        loadVideo(video.id);
      });
      card.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); loadVideo(video.id); }
      });

      libraryGrid.appendChild(card);
    });

    libraryCount.textContent = `${library.length} / ${MAX_VIDEOS}`;
    videoCountPill.textContent = `${library.length} / ${MAX_VIDEOS} throws loaded`;
    addMoreBtn.disabled = library.length >= MAX_VIDEOS;
    addMoreBtn.style.opacity = library.length >= MAX_VIDEOS ? 0.5 : 1;
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  /* ------------------------------------------------------------------ *
   * Fetching / uploading
   * ------------------------------------------------------------------ */

  async function fetchLibrary() {
    const res = await fetch("/api/videos");
    library = await res.json();
    renderLibrary();
    if (library.length > 0) {
      workspace.hidden = false;
      if (!activeVideoId) loadVideo(library[0].id);
    }
  }

  async function uploadFiles(fileList) {
    const files = Array.from(fileList);
    if (files.length === 0) return;

    const remaining = MAX_VIDEOS - library.length;
    if (remaining <= 0) {
      setUploadFeedback(`You already have the maximum of ${MAX_VIDEOS} throws. Delete one to add another.`, "error");
      return;
    }

    const toSend = files.slice(0, remaining);
    if (files.length > remaining) {
      setUploadFeedback(`Only ${remaining} more video(s) can be added — uploading the first ${remaining}.`, "error");
    } else {
      setUploadFeedback("Uploading...", null);
    }

    const formData = new FormData();
    toSend.forEach((f) => formData.append("videos", f));

    uploadBtn.disabled = true;
    try {
      const res = await fetch("/api/upload", { method: "POST", body: formData });
      const data = await res.json();

      if (data.created && data.created.length > 0) {
        setUploadFeedback(`${data.created.length} throw(s) uploaded.`, "success");
        showToast(`${data.created.length} video(s) added to your library.`);
      }
      if (data.errors && data.errors.length > 0) {
        setUploadFeedback(data.errors.join(" "), "error");
      }
      if ((!data.created || data.created.length === 0) && (!data.errors || data.errors.length === 0)) {
        setUploadFeedback(data.error || "Upload failed.", "error");
      }

      await fetchLibrary();
      workspace.hidden = false;
      if (data.created && data.created.length > 0) {
        loadVideo(data.created[data.created.length - 1].id);
        workspace.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    } catch (err) {
      setUploadFeedback("Upload failed — check your connection and try again.", "error");
    } finally {
      uploadBtn.disabled = false;
      fileInput.value = "";
    }
  }

  /* ------------------------------------------------------------------ *
   * Add from YouTube
   * ------------------------------------------------------------------ */

  function setYoutubeStatus(message, kind) {
    youtubeStatus.textContent = message || "";
    youtubeStatus.classList.remove("is-error", "is-success");
    if (kind) youtubeStatus.classList.add(kind === "error" ? "is-error" : "is-success");
  }

  function openYoutubeModal() {
    if (library.length >= MAX_VIDEOS) {
      showToast(`You already have the maximum of ${MAX_VIDEOS} throws. Delete one to add another.`);
      return;
    }
    youtubeInput.value = "";
    setYoutubeStatus("", null);
    youtubeModal.hidden = false;
    youtubeInput.focus();
  }

  function closeYoutubeModal() {
    youtubeModal.hidden = true;
  }

  async function addFromYoutube() {
    const url = youtubeInput.value.trim();
    if (!url) { setYoutubeStatus("Paste a YouTube URL first.", "error"); return; }

    youtubeConfirmBtn.disabled = true;
    youtubeInput.disabled = true;
    setYoutubeStatus("Downloading — this can take a little while for longer clips…", null);

    try {
      const res = await fetch("/api/youtube", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      const data = await res.json();

      if (!res.ok || !data.created || data.created.length === 0) {
        setYoutubeStatus(data.error || "Couldn't add that video.", "error");
        return;
      }

      closeYoutubeModal();
      showToast("Video added from YouTube.");
      await fetchLibrary();
      workspace.hidden = false;
      loadVideo(data.created[0].id);
      workspace.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      setYoutubeStatus("Request failed — check your connection and try again.", "error");
    } finally {
      youtubeConfirmBtn.disabled = false;
      youtubeInput.disabled = false;
    }
  }

  youtubeBtn.addEventListener("click", (e) => { e.stopPropagation(); openYoutubeModal(); });
  youtubeMoreBtn.addEventListener("click", openYoutubeModal);
  youtubeCancelBtn.addEventListener("click", closeYoutubeModal);
  youtubeConfirmBtn.addEventListener("click", addFromYoutube);
  youtubeModal.addEventListener("click", (e) => { if (e.target === youtubeModal) closeYoutubeModal(); });
  youtubeInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); addFromYoutube(); }
    if (e.key === "Escape") closeYoutubeModal();
  });

  /* ------------------------------------------------------------------ *
   * Trim toolbar — cut a shorter clip out of a longer video using the
   * player's own playhead to mark the range (e.g. isolating one player's
   * throws out of a recording shared between two people)
   * ------------------------------------------------------------------ */

  function setTrimStatus(message, kind) {
    trimStatusEl.textContent = message || "";
    trimStatusEl.classList.remove("is-error", "is-success");
    if (kind) trimStatusEl.classList.add(kind === "error" ? "is-error" : "is-success");
  }

  function updateTrimUI() {
    trimStartReadout.textContent = trimStart != null ? formatTime(trimStart) : "--:--.--";
    trimEndReadout.textContent = trimEnd != null ? formatTime(trimEnd) : "--:--.--";
    trimCreateBtn.disabled = trimBusy || !(trimStart != null && trimEnd != null && trimEnd > trimStart);
  }

  function resetTrimMarks() {
    trimStart = null;
    trimEnd = null;
    setTrimStatus("", null);
    updateTrimUI();
  }

  trimMarkStartBtn.addEventListener("click", () => {
    if (!activeVideo()) { showToast("Load a throw first."); return; }
    trimStart = videoEl.currentTime;
    setTrimStatus("", null);
    updateTrimUI();
  });

  trimMarkEndBtn.addEventListener("click", () => {
    if (!activeVideo()) { showToast("Load a throw first."); return; }
    trimEnd = videoEl.currentTime;
    setTrimStatus("", null);
    updateTrimUI();
  });

  trimCreateBtn.addEventListener("click", async () => {
    const video = activeVideo();
    if (!video || trimStart == null || trimEnd == null || trimEnd <= trimStart) return;

    if (library.length >= MAX_VIDEOS) {
      setTrimStatus(`You already have the maximum of ${MAX_VIDEOS} throws. Delete one to add another.`, "error");
      return;
    }

    trimBusy = true;
    updateTrimUI();
    setTrimStatus("Trimming — this takes a few seconds…", null);

    try {
      const res = await fetch(`/api/videos/${video.id}/trim`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ start: trimStart, end: trimEnd }),
      });
      const data = await res.json();

      if (!res.ok || !data.created || data.created.length === 0) {
        setTrimStatus(data.error || "Couldn't trim that video.", "error");
        return;
      }

      showToast("Trimmed clip added to your library.");
      await fetchLibrary();
      loadVideo(data.created[0].id);
    } catch (err) {
      setTrimStatus("Request failed — check your connection and try again.", "error");
    } finally {
      trimBusy = false;
      updateTrimUI();
    }
  });

  /* ------------------------------------------------------------------ *
   * Analyze by AI — a full written coaching take from an LLM, since the
   * heuristic scores above are an approximate formula, not a certified
   * rating. Reuses the same provider + API key storage as Ask AI's
   * "AI Chat" mode (see openApiKeyModal above).
   * ------------------------------------------------------------------ */

  function showAiAnalysisResult(text) {
    aiAnalysisResult.hidden = false;
    aiAnalysisResult.textContent = text;
    aiAnalysisResult.classList.remove("is-error");
  }

  function showAiAnalysisError(message) {
    aiAnalysisResult.hidden = false;
    aiAnalysisResult.textContent = message;
    aiAnalysisResult.classList.add("is-error");
  }

  async function runAiAnalysis() {
    const video = activeVideo();
    if (!video || !video.analysis) { showToast("Analyze this throw first."); return; }

    const provider = getStoredProvider();
    const apiKey = getStoredApiKey(provider);
    if (!apiKey) { openApiKeyModal(runAiAnalysis); return; }

    aiAnalysisBusy = true;
    aiAnalyzeBtn.disabled = true;
    aiAnalysisResult.hidden = false;
    aiAnalysisResult.classList.remove("is-error");
    aiAnalysisResult.textContent = "Thinking…";

    try {
      const res = await fetch(`/api/ai-analyze/${video.id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider, api_key: apiKey }),
      });
      const data = await res.json();

      if (!res.ok) {
        showAiAnalysisError(data.error || "Couldn't generate an AI analysis.");
        if (/api key/i.test(data.error || "")) openApiKeyModal(runAiAnalysis);
      } else {
        showAiAnalysisResult(data.result);
      }
    } catch (err) {
      showAiAnalysisError("Request failed — check your connection and try again.");
    } finally {
      aiAnalysisBusy = false;
      aiAnalyzeBtn.disabled = false;
    }
  }

  aiAnalyzeBtn.addEventListener("click", runAiAnalysis);

  async function deleteVideo(id) {
    const video = library.find((v) => v.id === id);
    if (!video) return;
    if (!confirm(`Delete "${video.display_name}"? This can't be undone.`)) return;

    await fetch(`/api/videos/${id}`, { method: "DELETE" });
    if (activeVideoId === id) {
      activeVideoId = null;
      videoEl.removeAttribute("src");
      stageEmpty.hidden = false;
      clearInfoPanel();
      clearAnalysisPanel();
    }
    await fetchLibrary();
    showToast("Throw deleted.");
  }

  async function renameVideo(id, newName) {
    const res = await fetch(`/api/videos/${id}/rename`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ display_name: newName }),
    });
    if (res.ok) {
      await fetchLibrary();
      showToast("Throw renamed.");
    }
  }

  /* ------------------------------------------------------------------ *
   * Rename modal
   * ------------------------------------------------------------------ */

  function openRenameModal(id) {
    const video = library.find((v) => v.id === id);
    if (!video) return;
    renameTargetId = id;
    renameInput.value = video.display_name;
    renameModal.hidden = false;
    renameInput.focus();
    renameInput.select();
  }

  function closeRenameModal() {
    renameModal.hidden = true;
    renameTargetId = null;
  }

  renameCancel.addEventListener("click", closeRenameModal);
  renameModal.addEventListener("click", (e) => { if (e.target === renameModal) closeRenameModal(); });
  renameConfirm.addEventListener("click", () => {
    const value = renameInput.value.trim();
    if (value && renameTargetId) renameVideo(renameTargetId, value);
    closeRenameModal();
  });
  renameInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") renameConfirm.click();
    if (e.key === "Escape") closeRenameModal();
  });

  /* ------------------------------------------------------------------ *
   * Player: loading a video + info panel
   * ------------------------------------------------------------------ */

  function clearInfoPanel() {
    infoFilename.textContent = "—";
    infoResolution.textContent = "—";
    infoFps.textContent = "—";
    infoDuration.textContent = "—";
    infoSize.textContent = "—";
  }

  /* ------------------------------------------------------------------ *
   * Throw Analysis panel (Version 2: real pose-detection results)
   * ------------------------------------------------------------------ */

  const SCORE_CLASS_BY_LABEL = {
    "Excellent": "score--excellent",
    "Very Good": "score--very-good",
    "Good": "score--good",
    "Needs Work": "score--needs-work",
  };

  function scoreClass(label) {
    return SCORE_CLASS_BY_LABEL[label] || "";
  }

  function clearAnalysisPanel() {
    [analysisElbow, analysisElbowDirection, analysisWrist, analysisRelease, analysisOverall].forEach((el) => {
      el.textContent = "––";
      el.className = el === analysisOverall ? "placeholder placeholder--score" : "placeholder";
    });
    analysisStatusPill.textContent = "Not analyzed";
    analysisStatusPill.className = "pill pill--soon";
    analysisNote.textContent = 'Click "Start Analysis" below to run AI pose detection on this throw.';
    releaseMarker.hidden = true;
    comparisonSummary.hidden = true;
    throwPictograph.hidden = true;
    throwPictograph.innerHTML = "";
    selectedThrowIndex = 0;
    resetCoachLog();
    resetAiAnalysis();
  }

  function resetAiAnalysis() {
    aiAnalysisResult.hidden = true;
    aiAnalysisResult.textContent = "";
    aiAnalysisResult.classList.remove("is-error");
  }

  function renderThrowDetail(throw_) {
    const elbow = throw_.elbow;
    const wrist = throw_.wrist;
    const overall = throw_.overall;

    analysisElbow.textContent = `${elbow.score}% — ${elbow.label}`;
    analysisElbow.className = `mono ${scoreClass(elbow.label)}`;

    analysisElbowDirection.textContent = elbow.direction ? elbow.direction.summary : "Not enough data";
    analysisElbowDirection.className = "mono";

    analysisWrist.textContent = `${wrist.score}% — ${wrist.label}`;
    analysisWrist.className = `mono ${scoreClass(wrist.label)}`;

    analysisRelease.textContent = `Frame ${throw_.release_frame} (${throw_.release_time.toFixed(2)}s)`;
    analysisRelease.className = "mono";

    analysisOverall.textContent = `${overall.score}%`;
    analysisOverall.className = `placeholder--score ${scoreClass(overall.label)}`;

    analysisStatusPill.textContent = overall.label;
    analysisStatusPill.className = `pill ${scoreClass(overall.label)}`;

    const video = activeVideo();
    const duration = (video && video.duration) || videoEl.duration || 0;
    if (duration > 0) {
      const pct = Math.min(100, (throw_.release_time / duration) * 100);
      releaseMarker.style.left = `${pct}%`;
      releaseMarker.title = `Throw #${throw_.throw_number} release (${throw_.release_time.toFixed(2)}s)`;
      releaseMarker.hidden = false;
    }
  }

  function selectThrow(index) {
    const video = activeVideo();
    if (!video || !video.analysis) return;
    selectedThrowIndex = Math.max(0, Math.min(index, video.analysis.throws.length - 1));
    renderThrowDetail(video.analysis.throws[selectedThrowIndex]);
    throwPictograph.querySelectorAll(".pictograph__bar").forEach((bar, i) => {
      bar.classList.toggle("is-selected", i === selectedThrowIndex);
      bar.setAttribute("aria-selected", i === selectedThrowIndex ? "true" : "false");
    });
  }

  function renderPictograph(analysis) {
    throwPictograph.innerHTML = "";
    throwPictograph.hidden = analysis.throws.length < 2;

    analysis.throws.forEach((throw_, i) => {
      const bar = document.createElement("button");
      bar.type = "button";
      bar.className = `pictograph__bar ${scoreClass(throw_.overall.label)}`;
      bar.setAttribute("role", "option");
      bar.setAttribute("aria-selected", "false");
      bar.title = `Throw #${throw_.throw_number}: ${throw_.overall.score}% (${throw_.overall.label})`;
      bar.innerHTML = `
        <span class="pictograph__value">${throw_.overall.score}</span>
        <span class="pictograph__fill" style="height:${Math.max(4, throw_.overall.score)}%"></span>
        <span class="pictograph__label">#${throw_.throw_number}</span>
      `;
      bar.addEventListener("click", () => selectThrow(i));
      throwPictograph.appendChild(bar);
    });
  }

  function renderAnalysis(analysis) {
    const c = analysis.comparison;
    const armLabel = analysis.arm === "left" ? "left arm" : "right arm";
    comparisonSummary.hidden = false;
    comparisonSummary.textContent = analysis.throw_count > 1
      ? `${analysis.throw_count} throws detected (${armLabel}) — average ${c.average_score}% (${c.average_label}), ` +
        `best #${c.best_throw_number}, consistency ${c.consistency_score}% (${c.consistency_label}).`
      : `1 throw detected in this clip (${armLabel}).`;

    renderPictograph(analysis);

    const bestIndex = analysis.throws.findIndex((t) => t.throw_number === c.best_throw_number);
    selectThrow(bestIndex >= 0 ? bestIndex : 0);

    analysisNote.textContent = `Based on ${analysis.frames_with_pose} tracked frames across ${analysis.throw_count} throw(s).`;
  }

  function loadVideo(id) {
    const video = library.find((v) => v.id === id);
    if (!video) return;

    activeVideoId = id;
    videoEl.src = video.video_url;
    videoEl.playbackRate = parseFloat(speedSlider.value);
    stageEmpty.hidden = true;

    infoFilename.textContent = video.display_name;
    infoResolution.textContent = `${video.width} × ${video.height}`;
    infoFps.textContent = video.fps ? video.fps.toFixed(2) : "—";
    infoDuration.textContent = `${video.duration.toFixed(1)} sec`;
    infoSize.textContent = video.file_size_readable;

    totalTimeEl.textContent = formatTime(video.duration);
    updateFrameReadout(0);
    updateTimelineUI(0);

    resetCoachLog();
    if (video.analysis) {
      setSelectedArm(video.analysis.arm);
      renderAnalysis(video.analysis);
    } else {
      setSelectedArm("right");
      clearAnalysisPanel();
    }

    resetAiAnalysis();
    if (video.ai_analysis && video.ai_analysis.text) {
      showAiAnalysisResult(video.ai_analysis.text);
    }

    resetTrimMarks();
    renderLibrary();
  }

  function updateFrameReadout(currentTime) {
    const video = activeVideo();
    const fps = video && video.fps ? video.fps : 30;
    const totalFrames = video ? video.frame_count : 0;
    const currentFrame = Math.min(totalFrames, Math.round(currentTime * fps));
    frameReadout.textContent = `Frame ${currentFrame} / ${totalFrames}`;
  }

  function updateTimelineUI(currentTime) {
    const video = activeVideo();
    const duration = (video && video.duration) || videoEl.duration || 0;
    const pct = duration > 0 ? Math.min(100, (currentTime / duration) * 100) : 0;
    timelineProgress.style.width = `${pct}%`;
    timelineHandle.style.left = `${pct}%`;
    currentTimeEl.textContent = formatTime(currentTime);
  }

  /* ------------------------------------------------------------------ *
   * Transport controls
   * ------------------------------------------------------------------ */

  function togglePlay() {
    if (!videoEl.src) return;
    if (videoEl.paused) videoEl.play(); else videoEl.pause();
  }

  playBtn.addEventListener("click", togglePlay);

  videoEl.addEventListener("play", () => {
    playIcon.innerHTML = '<rect x="6" y="5" width="4" height="14" rx="1"></rect><rect x="14" y="5" width="4" height="14" rx="1"></rect>';
  });
  videoEl.addEventListener("pause", () => {
    playIcon.innerHTML = '<path d="M8 5v14l11-7z"></path>';
  });

  stopBtn.addEventListener("click", () => {
    if (!videoEl.src) return;
    videoEl.pause();
    videoEl.currentTime = 0;
  });

  prevFrameBtn.addEventListener("click", () => {
    if (!videoEl.src) return;
    videoEl.pause();
    const step = frameDuration(activeVideo());
    videoEl.currentTime = Math.max(0, videoEl.currentTime - step);
  });

  nextFrameBtn.addEventListener("click", () => {
    if (!videoEl.src) return;
    videoEl.pause();
    const step = frameDuration(activeVideo());
    const duration = videoEl.duration || Infinity;
    videoEl.currentTime = Math.min(duration, videoEl.currentTime + step);
  });

  videoEl.addEventListener("timeupdate", () => {
    if (isScrubbing) return;
    updateTimelineUI(videoEl.currentTime);
    updateFrameReadout(videoEl.currentTime);
  });

  videoEl.addEventListener("loadedmetadata", () => {
    if (!activeVideo()) totalTimeEl.textContent = formatTime(videoEl.duration);
  });

  /* Fullscreen */

  fullscreenBtn.addEventListener("click", () => {
    const stage = document.querySelector(".player-card__stage");
    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else if (stage.requestFullscreen) {
      stage.requestFullscreen();
    }
  });

  /* Volume */

  muteBtn.addEventListener("click", () => {
    videoEl.muted = !videoEl.muted;
    updateVolumeIcon();
  });

  volumeSlider.addEventListener("input", () => {
    videoEl.volume = parseFloat(volumeSlider.value);
    videoEl.muted = videoEl.volume === 0;
    updateVolumeIcon();
  });

  function updateVolumeIcon() {
    const icon = document.getElementById("volume-icon");
    if (videoEl.muted || videoEl.volume === 0) {
      icon.innerHTML = '<path d="M4 9v6h4l5 5V4L8 9H4z"></path><path d="M18 9l-5 6M13 9l5 6" stroke="currentColor" stroke-width="1.8" fill="none" stroke-linecap="round"></path>';
    } else {
      icon.innerHTML = '<path d="M4 9v6h4l5 5V4L8 9H4z"></path><path d="M16 8.5a4.5 4.5 0 0 1 0 7" stroke="currentColor" stroke-width="1.8" fill="none" stroke-linecap="round"></path>';
    }
  }

  /* ------------------------------------------------------------------ *
   * Timeline scrubbing
   * ------------------------------------------------------------------ */

  function seekFromClientX(clientX) {
    const rect = timelineTrack.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
    const duration = videoEl.duration || (activeVideo() ? activeVideo().duration : 0);
    const time = ratio * duration;
    updateTimelineUI(time);
    updateFrameReadout(time);
    return time;
  }

  timelineTrack.addEventListener("pointerdown", (e) => {
    if (!videoEl.src) return;
    isScrubbing = true;
    timelineTrack.setPointerCapture(e.pointerId);
    videoEl.currentTime = seekFromClientX(e.clientX);
  });

  timelineTrack.addEventListener("pointermove", (e) => {
    if (!isScrubbing) return;
    videoEl.currentTime = seekFromClientX(e.clientX);
  });

  timelineTrack.addEventListener("pointerup", (e) => {
    isScrubbing = false;
    timelineTrack.releasePointerCapture(e.pointerId);
  });

  /* ------------------------------------------------------------------ *
   * Playback speed (0.10x – 2.00x)
   * ------------------------------------------------------------------ */

  function applySpeed(value) {
    const speed = Math.min(2, Math.max(0.1, parseFloat(value)));
    videoEl.playbackRate = speed;
    speedValue.textContent = `${speed.toFixed(2)}×`;
    speedSlider.value = speed;

    speedPresets.querySelectorAll(".chip").forEach((chip) => {
      chip.classList.toggle("is-active", parseFloat(chip.dataset.speed) === speed);
    });
  }

  speedSlider.addEventListener("input", () => applySpeed(speedSlider.value));

  speedPresets.addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (!chip) return;
    applySpeed(chip.dataset.speed);
  });

  /* ------------------------------------------------------------------ *
   * Throwing arm selector
   * ------------------------------------------------------------------ */

  function setSelectedArm(arm) {
    selectedArm = arm === "left" ? "left" : "right";
    armSelect.querySelectorAll(".chip--arm").forEach((chip) => {
      const active = chip.dataset.arm === selectedArm;
      chip.classList.toggle("is-active", active);
      chip.setAttribute("aria-checked", active ? "true" : "false");
    });
  }

  armSelect.addEventListener("click", (e) => {
    const chip = e.target.closest(".chip--arm");
    if (!chip) return;
    setSelectedArm(chip.dataset.arm);
  });

  /* ------------------------------------------------------------------ *
   * Analysis (Version 2: MediaPipe pose detection on the server)
   * ------------------------------------------------------------------ */

  analyzeBtn.addEventListener("click", async () => {
    if (!activeVideoId) {
      showToast("Select a throw first.");
      return;
    }
    const targetId = activeVideoId;
    const arm = selectedArm;
    analyzeBtn.disabled = true;
    analysisStatusPill.textContent = "Analyzing…";
    analysisStatusPill.className = "pill pill--soon";
    analysisNote.textContent = "Running AI pose detection — this can take a little while for longer clips.";

    try {
      const res = await fetch(`/api/analyze/${targetId}?arm=${encodeURIComponent(arm)}`, { method: "POST" });
      const data = await res.json();

      if (!res.ok || data.status === "error") {
        analysisNote.textContent = data.message || "Analysis failed. Try a clearer video.";
        analysisStatusPill.textContent = "Failed";
        analysisStatusPill.className = "pill";
        showToast(data.message || "Analysis failed.");
        return;
      }

      const entry = library.find((v) => v.id === targetId);
      if (entry) entry.analysis = data.analysis;

      if (targetId === activeVideoId) renderAnalysis(data.analysis);
      showToast(data.cached ? "Showing cached analysis." : "Analysis complete.");
    } catch (err) {
      analysisNote.textContent = "Analysis failed — check your connection and try again.";
      analysisStatusPill.textContent = "Failed";
      analysisStatusPill.className = "pill";
      showToast("Analysis failed.");
    } finally {
      analyzeBtn.disabled = false;
    }
  });

  /* ------------------------------------------------------------------ *
   * Ask AI (coaching Q&A over the computed analysis)
   * ------------------------------------------------------------------ */

  function resetCoachLog() {
    coachLog.innerHTML = "";
    coachLogEmpty.textContent = 'Analyze a throw, then ask about your elbow, wrist, direction, or how your throws compare.';
    coachLog.appendChild(coachLogEmpty);
  }

  function appendCoachMessage(role, text) {
    if (coachLogEmpty.parentNode) coachLogEmpty.remove();
    const bubble = document.createElement("div");
    bubble.className = `coach-msg coach-msg--${role}`;
    bubble.textContent = text;
    coachLog.appendChild(bubble);
    coachLog.scrollTop = coachLog.scrollHeight;
  }

  /* ---- AI provider + API key (AI Chat mode — bring your own key) ----
   * This app is used by other people, not just its developer, and different
   * visitors already have keys for different providers — so AI Chat lets
   * each person pick Claude, GPT, or Gemini and supply their own key rather
   * than sharing one server-side key for one provider. The provider choice
   * and each provider's key live only in this browser's localStorage and
   * ride along with each /api/coach request; the server never stores them
   * (see ai_coach.py — used for one API call and discarded). */

  const PROVIDERS = {
    anthropic: { label: "Anthropic (Claude)", short: "Claude", keyPlaceholder: "sk-ant-...", consoleUrl: "https://console.anthropic.com/settings/keys" },
    openai: { label: "OpenAI (GPT)", short: "GPT", keyPlaceholder: "sk-...", consoleUrl: "https://platform.openai.com/api-keys" },
    gemini: { label: "Google (Gemini)", short: "Gemini", keyPlaceholder: "AIza...", consoleUrl: "https://aistudio.google.com/apikey" },
  };
  const PROVIDER_STORAGE_KEY = "darts_ai_provider";
  const keyStorageKey = (provider) => `darts_api_key_${provider}`;

  function getStoredProvider() {
    try {
      const p = localStorage.getItem(PROVIDER_STORAGE_KEY);
      return PROVIDERS[p] ? p : "anthropic";
    } catch (err) { return "anthropic"; }
  }

  function setStoredProvider(provider) {
    try { localStorage.setItem(PROVIDER_STORAGE_KEY, provider); } catch (err) { /* ignore */ }
  }

  function getStoredApiKey(provider) {
    try { return localStorage.getItem(keyStorageKey(provider)) || ""; } catch (err) { return ""; }
  }

  function setStoredApiKey(provider, key) {
    try {
      if (key) localStorage.setItem(keyStorageKey(provider), key);
      else localStorage.removeItem(keyStorageKey(provider));
    } catch (err) {
      // Storage unavailable (private browsing, etc.) — key just won't persist across reloads.
    }
  }

  function updateApiKeyLink() {
    const provider = getStoredProvider();
    const short = PROVIDERS[provider].short;
    const label = getStoredApiKey(provider) ? `Using your saved ${short} key · Change` : "Set your API key";

    if (coachMode !== "llm") { apiKeyChangeBtn.hidden = true; } else {
      apiKeyChangeBtn.hidden = false;
      apiKeyChangeBtn.textContent = label;
    }
    aiAnalysisKeyLink.textContent = label;
  }

  function setCoachModeUI(mode) {
    coachMode = mode;
    coachModeGroup.querySelectorAll(".coach-mode__btn").forEach((b) => {
      const active = b.dataset.mode === mode;
      b.classList.toggle("is-active", active);
      b.setAttribute("aria-checked", active ? "true" : "false");
    });
    updateApiKeyLink();
  }

  function syncApiKeyModalToProvider() {
    const provider = apiKeyProviderSelect.value;
    apiKeyInput.value = getStoredApiKey(provider);
    apiKeyInput.placeholder = PROVIDERS[provider].keyPlaceholder;
    apiKeyConsoleLink.href = PROVIDERS[provider].consoleUrl;
    apiKeyConsoleLink.textContent = PROVIDERS[provider].consoleUrl.replace(/^https?:\/\//, "").replace(/\/.*$/, "");
  }

  /* `onReady`, if given, runs once a key has been saved for the selected
   * provider — lets a caller (e.g. "Analyze by AI") kick the modal open,
   * then resume the action it was trying to do rather than making the
   * user click twice. */
  function openApiKeyModal(onReady) {
    pendingApiKeyCallback = onReady || null;
    apiKeyProviderSelect.value = getStoredProvider();
    syncApiKeyModalToProvider();
    apiKeyModal.hidden = false;
    apiKeyInput.focus();
  }

  function closeApiKeyModal() {
    apiKeyModal.hidden = true;
  }

  coachModeGroup.addEventListener("click", (e) => {
    const btn = e.target.closest(".coach-mode__btn");
    if (!btn) return;
    setCoachModeUI(btn.dataset.mode);
    if (btn.dataset.mode === "llm" && !getStoredApiKey(getStoredProvider())) {
      apiKeyModalFromCoachToggle = true;
      openApiKeyModal();
    }
  });

  apiKeyChangeBtn.addEventListener("click", () => openApiKeyModal());
  aiAnalysisKeyLink.addEventListener("click", () => openApiKeyModal());
  apiKeyProviderSelect.addEventListener("change", syncApiKeyModalToProvider);

  apiKeySaveBtn.addEventListener("click", () => {
    const provider = apiKeyProviderSelect.value;
    const key = apiKeyInput.value.trim();
    if (!key) { showToast("Enter a valid API key."); return; }
    setStoredProvider(provider);
    setStoredApiKey(provider, key);
    closeApiKeyModal();
    updateApiKeyLink();
    showToast(`${PROVIDERS[provider].short} key saved in this browser.`);

    apiKeyModalFromCoachToggle = false;
    const callback = pendingApiKeyCallback;
    pendingApiKeyCallback = null;
    if (callback) callback();
  });

  apiKeyClearBtn.addEventListener("click", () => {
    const provider = apiKeyProviderSelect.value;
    setStoredApiKey(provider, "");
    apiKeyInput.value = "";
    updateApiKeyLink();
    showToast(`Saved ${PROVIDERS[provider].short} key cleared.`);
  });

  apiKeyCancelBtn.addEventListener("click", () => {
    closeApiKeyModal();
    if (apiKeyModalFromCoachToggle && !getStoredApiKey(getStoredProvider())) setCoachModeUI("rule");
    apiKeyModalFromCoachToggle = false;
    pendingApiKeyCallback = null;
  });

  apiKeyModal.addEventListener("click", (e) => { if (e.target === apiKeyModal) apiKeyCancelBtn.click(); });

  coachForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = coachInput.value.trim();
    if (!question || coachBusy) return;

    const video = activeVideo();
    if (!video || !video.analysis) {
      showToast("Analyze this throw first.");
      return;
    }

    appendCoachMessage("user", question);
    coachInput.value = "";
    coachBusy = true;
    coachSend.disabled = true;

    try {
      const payload = { question, mode: coachMode };
      if (coachMode === "llm") {
        const provider = getStoredProvider();
        payload.provider = provider;
        payload.api_key = getStoredApiKey(provider);
      }

      const res = await fetch(`/api/coach/${video.id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) {
        appendCoachMessage("error", data.error || "Couldn't get an answer.");
        if (coachMode === "llm" && /api key/i.test(data.error || "")) openApiKeyModal();
      } else {
        appendCoachMessage("assistant", data.answer);
      }
    } catch (err) {
      appendCoachMessage("error", "Request failed — check your connection and try again.");
    } finally {
      coachBusy = false;
      coachSend.disabled = false;
    }
  });

  /* ------------------------------------------------------------------ *
   * Upload wiring: dropzone, file picker, drag & drop
   * ------------------------------------------------------------------ */

  dropzone.addEventListener("click", (e) => {
    if (e.target.closest("#upload-btn") || e.target.closest("#record-btn") || e.target.closest("#youtube-btn")) return; // buttons have their own handlers
    fileInput.click();
  });
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });

  uploadBtn.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
  addMoreBtn.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", () => uploadFiles(fileInput.files));

  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("is-dragover");
    });
  });
  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("is-dragover");
    });
  });
  dropzone.addEventListener("drop", (e) => {
    if (e.dataTransfer && e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
  });

  /* ------------------------------------------------------------------ *
   * Record a throw directly from the camera (getUserMedia + MediaRecorder),
   * then hand the recorded clip to the same uploadFiles() flow as a picked
   * file — the backend doesn't need to know a video came from a webcam
   * instead of a file picker.
   * ------------------------------------------------------------------ */

  function pickRecorderMimeType() {
    const candidates = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
    if (typeof MediaRecorder === "undefined") return "";
    return candidates.find((t) => MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(t)) || "";
  }

  function resetRecordUI() {
    recordedChunks = [];
    recordedBlob = null;
    recordPreview.srcObject = mediaStream || null;
    recordPreview.src = "";
    recordPreview.muted = true;
    recordPreview.controls = false;
    recordIndicator.hidden = true;
    recordStartBtn.hidden = false;
    recordStopBtn.hidden = true;
    recordRetakeBtn.hidden = true;
    recordUseBtn.hidden = true;
    recordHint.textContent = mediaStream
      ? "Frame your throw, then hit Record."
      : "Allow camera access, frame your throw, then hit Record.";
  }

  function startRecordTimer() {
    recordStartedAt = Date.now();
    recordTimerEl.textContent = "00:00";
    recordTimerInterval = setInterval(() => {
      const secs = Math.floor((Date.now() - recordStartedAt) / 1000);
      recordTimerEl.textContent = formatTime(secs).slice(0, 5);
    }, 250);
  }

  function stopRecordTimer() {
    clearInterval(recordTimerInterval);
    recordTimerInterval = null;
  }

  async function openRecordModal() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showToast("This browser doesn't support camera recording.");
      return;
    }
    recordModal.hidden = false;
    resetRecordUI();
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      recordPreview.srcObject = mediaStream;
      recordHint.textContent = "Frame your throw, then hit Record.";
    } catch (err) {
      recordHint.textContent = "Camera access was denied or unavailable — check your browser/OS camera permissions.";
    }
  }

  function closeRecordModal() {
    stopRecordTimer();
    if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
    mediaRecorder = null;
    if (mediaStream) {
      mediaStream.getTracks().forEach((track) => track.stop());
      mediaStream = null;
    }
    recordModal.hidden = true;
  }

  recordStartBtn.addEventListener("click", () => {
    if (!mediaStream) {
      showToast("Camera isn't ready yet.");
      return;
    }
    recordedChunks = [];
    const mimeType = pickRecorderMimeType();
    try {
      mediaRecorder = mimeType ? new MediaRecorder(mediaStream, { mimeType }) : new MediaRecorder(mediaStream);
    } catch (err) {
      showToast("Recording isn't supported in this browser.");
      return;
    }

    mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) recordedChunks.push(e.data); };
    mediaRecorder.onstop = () => {
      recordedBlob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || "video/webm" });
      recordPreview.srcObject = null;
      recordPreview.src = URL.createObjectURL(recordedBlob);
      recordPreview.muted = false;
      recordPreview.controls = true;
      recordPreview.play();
    };

    mediaRecorder.start();
    startRecordTimer();
    recordIndicator.hidden = false;
    recordStartBtn.hidden = true;
    recordStopBtn.hidden = false;
    recordHint.textContent = "Recording… hit Stop when the throw is done.";
  });

  recordStopBtn.addEventListener("click", () => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
    stopRecordTimer();
    recordIndicator.hidden = true;
    recordStopBtn.hidden = true;
    recordRetakeBtn.hidden = false;
    recordUseBtn.hidden = false;
    recordHint.textContent = "Preview your recording, then use it or retake.";
  });

  recordRetakeBtn.addEventListener("click", () => {
    resetRecordUI();
  });

  recordUseBtn.addEventListener("click", async () => {
    if (!recordedBlob) return;
    const file = new File([recordedBlob], `throw-recording-${Date.now()}.webm`, {
      type: recordedBlob.type || "video/webm",
    });
    closeRecordModal();
    await uploadFiles([file]);
  });

  recordCancelBtn.addEventListener("click", closeRecordModal);
  recordModal.addEventListener("click", (e) => { if (e.target === recordModal) closeRecordModal(); });

  recordBtn.addEventListener("click", openRecordModal);
  recordMoreBtn.addEventListener("click", openRecordModal);

  /* ------------------------------------------------------------------ *
   * Keyboard shortcuts (space to play/pause, arrows to step frames)
   * ------------------------------------------------------------------ */

  document.addEventListener("keydown", (e) => {
    if (["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
    if (!activeVideoId) return;

    if (e.code === "Space") { e.preventDefault(); togglePlay(); }
    if (e.code === "ArrowRight") { e.preventDefault(); nextFrameBtn.click(); }
    if (e.code === "ArrowLeft") { e.preventDefault(); prevFrameBtn.click(); }
  });

  /* ------------------------------------------------------------------ *
   * Init
   * ------------------------------------------------------------------ */

  clearInfoPanel();
  updateVolumeIcon();
  fetchLibrary();
})();

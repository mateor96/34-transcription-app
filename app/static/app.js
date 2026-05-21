    const dropZone      = document.getElementById('drop-zone');
    const fileInput     = document.getElementById('file-input');
    const transcribeBtn = document.getElementById('transcribe-btn');
    const progressSec   = document.getElementById('progress-section');
    const progressFill  = document.getElementById('progress-fill');
    const stageVerbEl   = document.getElementById('stage-verb');
    const stageDetailEl = document.getElementById('stage-detail');
    const stagePctEl    = document.getElementById('stage-pct');
    const resultSec     = document.getElementById('result-section');
    const audioPlayer   = document.getElementById('audio-player');
    const transcriptEl  = document.getElementById('transcript');
    const legendEl      = document.getElementById('speaker-legend');

    let selectedFile    = null;
    let currentJobId   = null;   // job UUID (live) or archive UUID
    let isLiveJob      = false;  // true = result still in memory, false = loaded from archive
    let speakerNames   = {};     // { SPEAKER_00: "Alice", … }
    let currentFilename = null;

    const SPEAKER_COLORS = ['s0','s1','s2','s3','s4'];

    function displayName(id) {
      return speakerNames[id] || id.replace('SPEAKER_', 'Speaker ');
    }

    // ── Upload ────────────────────────────────────────────────────────────────

    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', () => setFile(fileInput.files[0]));
    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
      setFile(e.dataTransfer.files[0]);
    });

    function setFile(file) {
      if (!file) return;
      selectedFile = file;
      dropZone.classList.add('has-file');
      dropZone.innerHTML = `<strong>${file.name}</strong><small>${(file.size / 1e6).toFixed(1)} MB</small>`;
      transcribeBtn.disabled = false;
    }

    transcribeBtn.addEventListener('click', async () => {
      if (!selectedFile) return;
      transcribeBtn.disabled = true;
      progressSec.style.display = 'block';
      resultSec.style.display   = 'none';
      setProgress(0, 'Uploading…');

      const form   = new FormData();
      form.append('file', selectedFile);
      const params = new URLSearchParams();
      const minS   = document.getElementById('min-speakers').value;
      const maxS   = document.getElementById('max-speakers').value;
      if (minS) params.set('min_speakers', minS);
      if (maxS) params.set('max_speakers', maxS);

      const res        = await fetch(`/transcribe?${params}`, { method: 'POST', body: form });
      const { job_id } = await res.json();
      currentJobId     = job_id;
      isLiveJob        = true;
      audioPlayer.src  = URL.createObjectURL(selectedFile);
      listenProgress(job_id);
    });

    // Whimsical verbs cycled while we wait — pure cosmetics, like Claude Code.
    const STAGE_VERBS = [
      'Tinkering', 'Pondering', 'Listening', 'Decoding', 'Transcribing',
      'Synthesizing', 'Conjuring', 'Untangling', 'Noodling', 'Computing',
      'Distilling', 'Aligning', 'Cogitating', 'Whittling', 'Brewing',
    ];
    let _verbTimer = null;
    let _verbIdx   = 0;

    function startVerbCycle() {
      stopVerbCycle();
      _verbIdx = Math.floor(Math.random() * STAGE_VERBS.length);
      stageVerbEl.textContent = STAGE_VERBS[_verbIdx] + '…';
      _verbTimer = setInterval(() => {
        _verbIdx = (_verbIdx + 1) % STAGE_VERBS.length;
        stageVerbEl.classList.add('is-fading');
        setTimeout(() => {
          stageVerbEl.textContent = STAGE_VERBS[_verbIdx] + '…';
          stageVerbEl.classList.remove('is-fading');
        }, 250);
      }, 2200);
    }

    function stopVerbCycle() {
      if (_verbTimer) { clearInterval(_verbTimer); _verbTimer = null; }
      stageVerbEl.classList.remove('is-fading');
    }

    function setProgress(pct, detail) {
      progressFill.style.width = pct + '%';
      stageDetailEl.textContent = detail || '';
      stagePctEl.textContent    = pct != null ? `${Math.round(pct)}%` : '';
    }

    function listenProgress(jobId) {
      startVerbCycle();
      const es = new EventSource(`/progress/${jobId}`);
      es.onmessage = e => {
        const data = JSON.parse(e.data);
        setProgress(data.pct ?? 0, data.message ?? data.stage);
        if (data.stage === 'done')  {
          es.close();
          stopVerbCycle();
          stageVerbEl.textContent = 'Done';
          loadLiveResult(jobId);
        }
        if (data.stage === 'error') {
          es.close();
          stopVerbCycle();
          stageVerbEl.textContent = 'Error';
          stageDetailEl.textContent = data.message;
        }
      };
    }

    async function loadLiveResult(jobId) {
      const res  = await fetch(`/result/${jobId}/json`);
      const data = await res.json();
      setResultTitle(selectedFile ? selectedFile.name : '');
      showSegments(data.segments);
      resultSec.style.display = 'block';
    }

    function setResultTitle(name) {
      currentFilename = name;
      document.getElementById('result-title').textContent = name;
    }

    // ── Shared render ─────────────────────────────────────────────────────────

    function showSegments(segments, savedNames = {}) {
      speakerNames = {};
      resetSummaryPanel();
      const seen = [];
      segments.forEach(s => { if (!seen.includes(s.speaker)) seen.push(s.speaker); });
      renderLegend(seen, savedNames);
      renderTranscript(segments);
    }

    let _saveNamesTimer = null;

    function saveNamesDebounced() {
      if (isLiveJob || !currentJobId) return;
      clearTimeout(_saveNamesTimer);
      _saveNamesTimer = setTimeout(() => {
        fetch(`/archive/${currentJobId}/names`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(speakerNames),
        });
      }, 600);
    }

    function renderLegend(speakers, savedNames = {}) {
      // Merge saved names into speakerNames
      Object.assign(speakerNames, savedNames);

      legendEl.innerHTML = speakers.map((id, i) => {
        const color = SPEAKER_COLORS[i % SPEAKER_COLORS.length];
        return `<div class="speaker-chip">
          <div class="chip-dot ${color}"></div>
          <input class="chip-input" data-id="${id}" value="${escapeHtml(displayName(id))}" placeholder="Name…">
        </div>`;
      }).join('');

      legendEl.querySelectorAll('.chip-input').forEach(input => {
        input.addEventListener('input', () => {
          const val = input.value.trim();
          speakerNames[input.dataset.id] = val || null;
          transcriptEl.querySelectorAll(`.${input.dataset.id} .turn-speaker`).forEach(el => {
            el.textContent = displayName(input.dataset.id);
          });
          saveNamesDebounced();
        });
        input.addEventListener('focus', () => input.select());
      });
    }

    function renderTranscript(segments) {
      transcriptEl.innerHTML = segments.map(seg => {
        const mins = Math.floor(seg.start / 60).toString().padStart(2, '0');
        const secs = Math.floor(seg.start % 60).toString().padStart(2, '0');
        return `<div class="turn ${seg.speaker}" data-start="${seg.start}">
          <div class="turn-meta">
            <div class="turn-speaker">${displayName(seg.speaker)}</div>
            <div class="turn-time">${mins}:${secs}</div>
          </div>
          <div class="turn-text">${escapeHtml(seg.text)}</div>
        </div>`;
      }).join('');

      transcriptEl.querySelectorAll('.turn').forEach(el => {
        el.addEventListener('click', () => {
          audioPlayer.currentTime = parseFloat(el.dataset.start);
          audioPlayer.play();
        });
      });
    }

    async function dlFile(fmt) {
      if (!currentJobId) return;
      const url = isLiveJob
        ? `/result/${currentJobId}/${fmt}`
        : `/archive/${currentJobId}/download/${fmt}`;
      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        let text = await res.text();
        Object.entries(speakerNames).forEach(([id, name]) => {
          if (name) text = text.replaceAll(id, name);
        });
        triggerDownload(new Blob([text], { type: 'text/plain' }), `transcript.${fmt}`);
      } catch (e) {
        alert('Download failed: ' + e.message);
      }
    }

    function copyTranscript() {
      const turns = [...transcriptEl.querySelectorAll('.turn')];
      if (!turns.length) return;
      const text = turns.map(el => {
        const speaker = el.querySelector('.turn-speaker').textContent;
        const time    = el.querySelector('.turn-time').textContent;
        const body    = el.querySelector('.turn-text').textContent.trim();
        return `[${speaker} ${time}]\n${body}`;
      }).join('\n\n');
      navigator.clipboard.writeText(text).then(() => {
        const btn = document.getElementById('copy-btn');
        const prev = btn.innerHTML;
        btn.innerHTML = '<svg width="11" height="11" viewBox="0 0 11 11" fill="none"><path d="M1.5 6l3 3 5-5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>Copied!';
        btn.style.color = 'var(--accent)';
        btn.style.borderColor = 'var(--accent)';
        setTimeout(() => { btn.innerHTML = prev; btn.style.color = ''; btn.style.borderColor = ''; }, 1800);
      });
    }

    function dlPDF() {
      if (!currentJobId) return;
      const rows = [...transcriptEl.querySelectorAll('.turn')].map(el => {
        const speaker = el.querySelector('.turn-speaker').textContent;
        const time    = el.querySelector('.turn-time').textContent;
        const text    = el.querySelector('.turn-text').textContent;
        return `<div class="turn">
          <div class="meta"><span class="spk">${speaker}</span><span class="ts">${time}</span></div>
          <div class="txt">${text}</div>
        </div>`;
      }).join('');

      const win = window.open('', '_blank');
      win.document.write(`<!DOCTYPE html><html><head><title>Transcript</title>
<style>
  body { font-family: Georgia, serif; max-width: 640px; margin: 2.5cm auto; font-size: 11pt; line-height: 1.65; color: #2d2a26; }
  h1 { font-weight: normal; font-size: 1.3rem; margin-bottom: 2rem; border-bottom: 1px solid #ccc; padding-bottom: 0.75rem; }
  .turn { display: grid; grid-template-columns: 100px 1fr; gap: 1rem; padding: 0.7rem 0; border-bottom: 1px solid #f0ece4; }
  .meta { padding-top: 0.1rem; }
  .spk { display: block; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; }
  .ts  { display: block; font-size: 0.72rem; color: #9a8f82; font-family: monospace; margin-top: 0.1rem; }
  .txt { font-size: 0.97rem; }
  @media print { body { margin: 1.5cm; } }
</style></head><body>
<h1>Transcript</h1>${rows}
<script>setTimeout(() => { window.print(); }, 400);<\/script>
</body></html>`);
      win.document.close();
    }

    function triggerDownload(blob, filename) {
      const url = URL.createObjectURL(blob);
      const a   = document.createElement('a');
      a.href    = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }

    function escapeHtml(s) {
      return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }

    // ── Tabs ──────────────────────────────────────────────────────────────────

    function switchTab(tab) {
      document.querySelectorAll('.tab').forEach((el, i) => {
        el.classList.toggle('active', (i === 0) === (tab === 'transcribe'));
      });
      document.getElementById('archive-view').style.display   = tab === 'archive'    ? 'block' : 'none';
      document.getElementById('upload-section').style.display = tab === 'transcribe' ? 'block' : 'none';
      progressSec.style.display = 'none';
      resultSec.style.display   = 'none';
      if (tab === 'archive') loadArchive();
    }

    // ── Archive ───────────────────────────────────────────────────────────────

    async function loadArchive() {
      const entries = await fetch('/archive').then(r => r.json());
      const el = document.getElementById('archive-list');
      if (!entries.length) {
        el.innerHTML = '<div class="archive-empty">No transcriptions yet.</div>';
        return;
      }
      el.innerHTML = entries.map(e => {
        const date = new Date(e.created_at).toLocaleDateString('de-DE', { day:'2-digit', month:'short', year:'numeric' });
        const dur     = fmtDuration(e.duration_s);
        const spk     = e.speaker_count === 1 ? '1 speaker' : `${e.speaker_count} speakers`;
        const sumBadge = e.has_summary ? ' &middot; <span style="color:var(--accent)">✦</span>' : '';
        return `<div class="archive-entry" onclick="loadArchiveEntry('${e.id}')">
          <div style="min-width:0">
            <div class="entry-name">${escapeHtml(e.filename)}</div>
            <div class="entry-meta">${dur} &middot; ${spk}${sumBadge}</div>
          </div>
          <div class="entry-right">
            <div class="entry-date">${date}</div>
            <button class="entry-rename-btn" title="Rename" onclick="startArchiveRename(event,'${e.id}',this)"><svg width="10" height="10" viewBox="0 0 12 12" fill="none"><path d="M8.5 1.5a1.414 1.414 0 0 1 2 2L3.5 10.5 1 11l.5-2.5 7-7z" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/></svg></button>
            <button class="entry-delete" title="Delete" onclick="deleteEntry(event,'${e.id}')">✕</button>
          </div>
        </div>`;
      }).join('');
    }

    async function loadArchiveEntry(id) {
      const entry  = await fetch(`/archive/${id}`).then(r => r.json());
      currentJobId = id;
      isLiveJob    = false;
      audioPlayer.src = entry.audio_ext ? `/archive/${id}/audio` : '';
      setResultTitle(entry.filename || '');
      showSegments(entry.segments, entry.speaker_names || {});
      document.getElementById('archive-view').style.display = 'none';
      document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
      resultSec.style.display = 'block';
      if (entry.summary) {
        document.getElementById('summary-section').style.display = 'block';
        renderSummaryText(entry.summary);
        document.getElementById('summarize-btn').textContent = '↻ Regenerate';
      }
    }

    async function deleteEntry(evt, id) {
      evt.stopPropagation();
      const entry = evt.target.closest('.archive-entry');
      const name  = entry?.querySelector('.entry-name')?.textContent?.trim() || 'this transcript';
      if (!confirm(`Delete "${name}"?\n\nThe transcript, summary, and audio file will be permanently removed.`)) return;
      await fetch(`/archive/${id}`, { method: 'DELETE' });
      loadArchive();
    }

    function startArchiveRename(evt, id, btn) {
      evt.stopPropagation();
      const nameEl = btn.closest('.archive-entry').querySelector('.entry-name');
      const prev   = nameEl.textContent;
      const input  = document.createElement('input');
      input.className = 'rename-input';
      input.value = prev;
      nameEl.replaceWith(input);
      input.focus(); input.select();
      let committed = false;
      async function commit() {
        if (committed) return; committed = true;
        const val = input.value.trim() || prev;
        const newEl = document.createElement('div');
        newEl.className = 'entry-name'; newEl.textContent = val;
        input.replaceWith(newEl);
        await fetch(`/archive/${id}/rename`, {
          method: 'PATCH', headers: {'Content-Type':'application/json'},
          body: JSON.stringify({ filename: val })
        });
        if (currentJobId === id) setResultTitle(val);
      }
      input.addEventListener('blur', commit);
      input.addEventListener('keydown', e => {
        if (e.key === 'Enter') { e.preventDefault(); input.blur(); }
        if (e.key === 'Escape') { input.value = prev; input.blur(); }
      });
    }

    function activateRenameTitle() {
      const span = document.getElementById('result-title');
      const prev = span.textContent;
      const row  = span.parentElement;
      const input = document.createElement('input');
      input.className = 'rename-input';
      input.style.fontFamily = 'Georgia, serif';
      input.style.fontSize   = '1.05rem';
      input.value = prev;
      span.replaceWith(input);
      input.focus(); input.select();
      let committed = false;
      async function commit() {
        if (committed) return; committed = true;
        const val = input.value.trim() || prev;
        const newSpan = document.createElement('span');
        newSpan.id = 'result-title'; newSpan.className = 'result-title';
        newSpan.textContent = val;
        input.replaceWith(newSpan);
        currentFilename = val;
        if (currentJobId) {
          await fetch(`/archive/${currentJobId}/rename`, {
            method: 'PATCH', headers: {'Content-Type':'application/json'},
            body: JSON.stringify({ filename: val })
          });
        }
      }
      input.addEventListener('blur', commit);
      input.addEventListener('keydown', e => {
        if (e.key === 'Enter') { e.preventDefault(); input.blur(); }
        if (e.key === 'Escape') { input.value = prev; input.blur(); }
      });
    }

    function fmtDuration(s) {
      if (!s) return '—';
      const m = Math.floor(s / 60), sec = Math.floor(s % 60);
      return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
    }

    // ── Input mode toggle ─────────────────────────────────────────────────────

    function switchInputMode(mode) {
      document.getElementById('mode-upload-btn').classList.toggle('active', mode === 'upload');
      document.getElementById('mode-record-btn').classList.toggle('active', mode === 'record');
      document.getElementById('upload-content').style.display  = mode === 'upload' ? 'block' : 'none';
      document.getElementById('record-content').style.display  = mode === 'record' ? 'block' : 'none';
      if (mode === 'record') populateDevices();
    }

    // ── Recording ─────────────────────────────────────────────────────────────

    let mediaRecorder   = null;
    let recordedChunks  = [];
    let recordingStream = null;
    let meterRaf        = null;
    let timerInterval   = null;
    let recordStartTime = null;
    let isRecording     = false;

    async function populateDevices() {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const inputs  = devices.filter(d => d.kind === 'audioinput');
        const sel     = document.getElementById('audio-device');
        sel.innerHTML = inputs.map(d =>
          `<option value="${d.deviceId}">${escapeHtml(d.label || 'Microphone ' + d.deviceId.slice(0,6))}</option>`
        ).join('');
      } catch (e) { console.warn('enumerateDevices:', e); }
    }

    async function toggleRecording() {
      isRecording ? stopRecording() : await startRecording();
    }

    async function startRecording() {
      const deviceId = document.getElementById('audio-device').value;
      try {
        const constraints = {
          audio: deviceId
            ? { deviceId: { exact: deviceId }, echoCancellation: false, noiseSuppression: false, autoGainControl: false }
            : { echoCancellation: false, noiseSuppression: false, autoGainControl: false }
        };
        recordingStream = await navigator.mediaDevices.getUserMedia(constraints);
      } catch (e) {
        alert('Kein Zugriff auf das Audiogerät: ' + e.message);
        return;
      }

      await populateDevices(); // re-enumerate with real labels now that permission is granted

      isRecording    = true;
      recordedChunks = [];
      mediaRecorder  = new MediaRecorder(recordingStream);
      mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recordedChunks.push(e.data); };
      mediaRecorder.onstop = onRecordingStop;
      mediaRecorder.start(1000);

      startMeter(recordingStream);
      startRecordTimer();

      const btn = document.getElementById('record-btn');
      btn.textContent = 'Stop & Transcribe';
      btn.classList.add('recording');
      document.getElementById('record-box').classList.add('active');
      document.getElementById('record-status-text').textContent = 'Recording…';
    }

    function stopRecording() {
      if (mediaRecorder?.state !== 'inactive') mediaRecorder.stop();
      recordingStream?.getTracks().forEach(t => t.stop());
      stopMeter();
      stopRecordTimer();
      isRecording = false;

      const btn = document.getElementById('record-btn');
      btn.textContent = 'Start Recording';
      btn.classList.remove('recording');
      document.getElementById('record-box').classList.remove('active');
      document.getElementById('record-status-text').textContent = 'Processing…';
    }

    function onRecordingStop() {
      const mime = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/ogg';
      const ext  = mime.includes('webm') ? 'webm' : 'ogg';
      const blob = new Blob(recordedChunks, { type: mime });
      const ts   = new Date().toISOString().slice(0,16).replace('T','_').replaceAll(':','-');
      const file = new File([blob], `Recording_${ts}.${ext}`, { type: mime });

      // Copy speaker hints to upload inputs, then hand off to existing pipeline
      document.getElementById('min-speakers').value = document.getElementById('rec-min-speakers').value;
      document.getElementById('max-speakers').value = document.getElementById('rec-max-speakers').value;
      switchInputMode('upload');
      setFile(file);
      transcribeBtn.click();
    }

    // ── Level meter ───────────────────────────────────────────────────────────

    function startMeter(stream) {
      const ctx    = new AudioContext();
      const node   = ctx.createAnalyser();
      node.fftSize = 256;
      ctx.createMediaStreamSource(stream).connect(node);
      const data   = new Uint8Array(node.frequencyBinCount);
      const fill   = document.getElementById('meter-fill');
      function tick() {
        node.getByteFrequencyData(data);
        const avg = data.reduce((a, b) => a + b, 0) / data.length;
        fill.style.width = Math.min(100, avg * 3) + '%';
        meterRaf = requestAnimationFrame(tick);
      }
      tick();
    }

    function stopMeter() {
      if (meterRaf) cancelAnimationFrame(meterRaf);
      meterRaf = null;
      const fill = document.getElementById('meter-fill');
      if (fill) fill.style.width = '0%';
    }

    // ── Record timer ──────────────────────────────────────────────────────────

    function startRecordTimer() {
      recordStartTime = Date.now();
      const el = document.getElementById('record-timer');
      timerInterval = setInterval(() => {
        const ms = Date.now() - recordStartTime;
        const h  = Math.floor(ms / 3600000).toString().padStart(2, '0');
        const m  = Math.floor((ms % 3600000) / 60000).toString().padStart(2, '0');
        const s  = Math.floor((ms % 60000) / 1000).toString().padStart(2, '0');
        el.textContent = `${h}:${m}:${s}`;
      }, 1000);
    }

    function stopRecordTimer() {
      clearInterval(timerInterval);
      timerInterval = null;
      document.getElementById('record-timer').textContent = '00:00:00';
    }

    // ── Settings ──────────────────────────────────────────────────────────────

    const PROVIDERS = [
      { id: 'lmstudio',  label: 'LM Studio' },
      { id: 'ollama',    label: 'Ollama' },
      { id: 'gemini',    label: 'Gemini' },
      { id: 'anthropic', label: 'Anthropic' },
      { id: 'openai',    label: 'OpenAI' },
    ];

    const LOCAL_PROVIDERS = new Set(['lmstudio', 'ollama']);
    const DEFAULT_URLS    = { lmstudio: 'http://localhost:1234', ollama: 'http://localhost:11434' };
    const PROMPT_STYLES = [
      { id: 'meeting',   label: 'Meeting' },
      { id: 'call',      label: 'Phone call' },
      { id: 'interview', label: 'Interview' },
      { id: 'lecture',   label: 'Lecture' },
      { id: 'custom',    label: 'Custom' },
    ];

    let _settingsProvider = 'lmstudio';
    let _savedApiKey = '';  // tracks whether a key is already saved (so we don't wipe it)
    let _settingsCfg = null;  // last-loaded server config — used to seed prompt style fields

    async function openSettings() {
      const overlay = document.getElementById('settings-overlay');
      overlay.style.display = 'flex';
      document.getElementById('test-result').textContent = '';
      document.getElementById('test-result').className = 'test-result';

      const cfg = await fetch('/settings').then(r => r.json());
      _settingsProvider = cfg.provider || 'lmstudio';
      _savedApiKey = cfg.api_key_set ? '__saved__' : '';
      _settingsCfg = cfg;

      renderProviderTiles(_settingsProvider);
      await renderSettingsFields(_settingsProvider, cfg);
      renderPromptStyleFields(cfg.prompt_style || 'meeting', cfg.custom_prompt || '');
    }

    function renderPromptStyleFields(style, customPrompt) {
      const container = document.getElementById('settings-prompt-section');
      if (!container) return;
      const opts = PROMPT_STYLES.map(p =>
        `<option value="${p.id}"${p.id === style ? ' selected' : ''}>${p.label}</option>`
      ).join('');
      container.innerHTML = `
        <div class="settings-field">
          <label>Summary style</label>
          <select id="sf-prompt-style" onchange="onPromptStyleChange()">${opts}</select>
        </div>
        <div class="settings-field" id="sf-custom-prompt-row" style="display:${style === 'custom' ? '' : 'none'}">
          <label>Custom prompt</label>
          <textarea id="sf-custom-prompt" rows="6" placeholder="Use {transcript} as a placeholder for the transcript text. If you omit it, the transcript is appended at the end.">${escapeHtml(customPrompt)}</textarea>
        </div>`;
    }

    function onPromptStyleChange() {
      const sel = document.getElementById('sf-prompt-style');
      const row = document.getElementById('sf-custom-prompt-row');
      if (sel && row) row.style.display = sel.value === 'custom' ? '' : 'none';
    }

    async function fetchModels(provider, baseUrl) {
      try {
        const res = await fetch('/models', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ provider, base_url: baseUrl || '' }),
        });
        return await res.json();
      } catch (e) {
        return { models: [], error: String(e) };
      }
    }

    function closeSettings() {
      document.getElementById('settings-overlay').style.display = 'none';
    }

    function closeSettingsIfOutside(e) {
      if (e.target === document.getElementById('settings-overlay')) closeSettings();
    }

    function renderProviderTiles(active) {
      document.getElementById('provider-tiles').innerHTML = PROVIDERS.map(p =>
        `<button class="provider-tile${p.id === active ? ' active' : ''}" onclick="selectProvider('${p.id}')">${p.label}</button>`
      ).join('');
    }

    async function renderSettingsFields(provider, cfg = {}) {
      const fields = document.getElementById('settings-fields');
      const isLocal = LOCAL_PROVIDERS.has(provider);
      const urlVal = cfg.base_url || DEFAULT_URLS[provider] || '';

      if (isLocal) {
        fields.innerHTML = `
          <div class="settings-field">
            <label>Base URL</label>
            <input type="text" id="sf-url" value="${escapeHtml(urlVal)}" placeholder="${DEFAULT_URLS[provider]}">
          </div>
          <div class="settings-field">
            <label>Model</label>
            <select id="sf-model" disabled><option>Loading…</option></select>
            <div class="model-hint" id="sf-model-hint"></div>
          </div>`;
        document.getElementById('sf-url').addEventListener('blur', () => populateModelDropdown(provider, cfg.model));
      } else {
        const keyPlaceholder = cfg.api_key_set ? '••••••••••••••••••••' : 'Enter API key';
        fields.innerHTML = `
          <div class="settings-field">
            <label>API Key</label>
            <div class="key-row">
              <input type="password" id="sf-key" placeholder="${keyPlaceholder}" autocomplete="off">
              <button class="eye-btn" onclick="toggleKeyVisibility()" id="eye-btn">👁</button>
            </div>
          </div>
          <div class="settings-field">
            <label>Model</label>
            <select id="sf-model" disabled><option>Loading…</option></select>
            <div class="model-hint" id="sf-model-hint"></div>
          </div>`;
      }
      document.getElementById('test-result').textContent = '';
      document.getElementById('test-result').className = 'test-result';

      await populateModelDropdown(provider, cfg.model);
    }

    async function populateModelDropdown(provider, savedModel) {
      const select = document.getElementById('sf-model');
      const hint   = document.getElementById('sf-model-hint');
      if (!select) return;
      const baseUrl = LOCAL_PROVIDERS.has(provider) ? (document.getElementById('sf-url')?.value || DEFAULT_URLS[provider]) : '';
      select.disabled = true;
      select.innerHTML = '<option>Loading…</option>';
      hint.textContent = '';
      hint.className = 'model-hint';

      const { models = [], error } = await fetchModels(provider, baseUrl);

      if (error) {
        select.innerHTML = '<option value="">— unavailable —</option>';
        hint.textContent = error;
        hint.className = 'model-hint error';
        return;
      }
      if (models.length === 0) {
        select.innerHTML = '<option value="">— no models found —</option>';
        if (provider === 'lmstudio') hint.textContent = 'Load a model in LM Studio and try again.';
        if (provider === 'ollama')   hint.textContent = 'Pull a model first, e.g. `ollama pull llama3.2:3b`.';
        return;
      }

      const options = models.map(m =>
        `<option value="${escapeHtml(m)}"${m === savedModel ? ' selected' : ''}>${escapeHtml(m)}</option>`
      );
      // If saved model is not in the fetched list, add it as a sticky option so it stays selected
      if (savedModel && !models.includes(savedModel)) {
        options.unshift(`<option value="${escapeHtml(savedModel)}" selected>${escapeHtml(savedModel)} (saved, not currently available)</option>`);
      }
      select.innerHTML = options.join('');
      select.disabled = false;
    }

    async function selectProvider(provider) {
      _settingsProvider = provider;
      _savedApiKey = '';
      renderProviderTiles(provider);
      await renderSettingsFields(provider);
    }

    function toggleKeyVisibility() {
      const inp = document.getElementById('sf-key');
      const btn = document.getElementById('eye-btn');
      if (!inp) return;
      inp.type = inp.type === 'password' ? 'text' : 'password';
      btn.textContent = inp.type === 'password' ? '👁' : '🙈';
    }

    function collectSettingsPayload() {
      const isLocal = LOCAL_PROVIDERS.has(_settingsProvider);
      const payload = { provider: _settingsProvider };
      if (isLocal) {
        payload.base_url = (document.getElementById('sf-url')?.value || '').trim();
        payload.model    = (document.getElementById('sf-model')?.value || '').trim();
        payload.api_key  = '';
      } else {
        const keyInput = document.getElementById('sf-key')?.value || '';
        payload.api_key = keyInput || (_savedApiKey === '__saved__' ? '__keep__' : '');
        payload.model   = (document.getElementById('sf-model')?.value || '').trim();
        payload.base_url = '';
      }
      payload.prompt_style  = document.getElementById('sf-prompt-style')?.value || 'meeting';
      payload.custom_prompt = document.getElementById('sf-custom-prompt')?.value || '';
      return payload;
    }

    async function testProvider() {
      const payload = collectSettingsPayload();
      const btn = document.getElementById('test-btn');
      const res = document.getElementById('test-result');
      btn.disabled = true;
      res.textContent = 'Testing…';
      res.className = 'test-result';

      const data = await fetch('/settings/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      }).then(r => r.json());

      btn.disabled = false;
      res.textContent = data.ok ? '✓ ' + data.message : '✗ ' + data.message;
      res.className = 'test-result ' + (data.ok ? 'ok' : 'error');
    }

    async function saveSettings() {
      const payload = collectSettingsPayload();
      // If api_key wasn't typed but one is already saved, omit the field so the
      // backend preserves it (POST /settings does partial-update fallback).
      if (payload.api_key === '__keep__') {
        delete payload.api_key;
      }
      await fetch('/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      _savedApiKey = (payload.api_key && payload.api_key !== '__keep__') ? '__saved__' : _savedApiKey;
      const btn = document.getElementById('settings-save-btn');
      const prev = btn.textContent;
      btn.textContent = 'Saved ✓';
      btn.style.background = '#3d6b50';
      btn.style.borderColor = '#3d6b50';
      setTimeout(() => {
        btn.textContent = prev;
        btn.style.background = '';
        btn.style.borderColor = '';
      }, 2000);
    }

    // ── Summarize ─────────────────────────────────────────────────────────────

    let _summaryEs = null;

    async function requestSummary() {
      if (!currentJobId) return;

      // Check provider is configured
      const cfg = await fetch('/settings').then(r => r.json());
      const summarySection  = document.getElementById('summary-section');
      const summaryContent  = document.getElementById('summary-content');
      const summaryLoading  = document.getElementById('summary-loading');
      const summaryError    = document.getElementById('summary-error');
      const summarizeBtn    = document.getElementById('summarize-btn');

      summarySection.style.display = 'block';
      document.getElementById('summary-chevron').classList.remove('collapsed');
      document.getElementById('summary-body').style.display = '';

      if (!cfg.provider) {
        summaryContent.innerHTML = '';
        summaryLoading.style.display = 'none';
        summaryError.style.display = 'block';
        summaryError.innerHTML = 'No AI provider configured. Open Settings (⚙ top right) to set one up.';
        return;
      }

      summaryError.style.display = 'none';
      summaryLoading.style.display = 'flex';
      document.getElementById('summary-stage-text').textContent = 'Summarizing…';
      summaryContent.innerHTML = '';
      summarizeBtn.disabled = true;

      if (_summaryEs) { _summaryEs.close(); _summaryEs = null; }

      let accumulated = '';
      _summaryEs = new EventSource(`/summarize/${currentJobId}`);
      _summaryEs.onmessage = e => {
        const data = JSON.parse(e.data);
        if (data.type === 'stage') {
          // Long-transcript map-reduce — show progress until the combine pass starts streaming.
          document.getElementById('summary-stage-text').textContent = data.message;
        }
        if (data.type === 'token') {
          accumulated += data.text;
          summaryContent.textContent = accumulated;
        }
        if (data.type === 'done') {
          _summaryEs.close(); _summaryEs = null;
          summaryLoading.style.display = 'none';
          renderSummaryText(data.summary);
          summarizeBtn.textContent = '↻ Regenerate';
          summarizeBtn.disabled = false;
        }
        if (data.type === 'error') {
          _summaryEs.close(); _summaryEs = null;
          summaryLoading.style.display = 'none';
          summaryContent.innerHTML = '';
          summaryError.style.display = 'block';
          summaryError.textContent = data.message;
          summarizeBtn.disabled = false;
        }
      };
      _summaryEs.onerror = () => {
        _summaryEs.close(); _summaryEs = null;
        summaryLoading.style.display = 'none';
        summaryError.style.display = 'block';
        summaryError.textContent = 'Connection error — is the server running?';
        summarizeBtn.disabled = false;
      };
    }

    function renderSummaryText(text) {
      const content = document.getElementById('summary-content');
      if (!text) { content.textContent = ''; return; }
      // Convert lines starting with - or • into an <ul>
      const lines = text.split('\n').filter(l => l.trim());
      const hasBullets = lines.some(l => /^[-•*]/.test(l.trim()));
      if (hasBullets) {
        const items = lines.map(l => {
          const trimmed = l.trim().replace(/^[-•*]\s*/, '');
          return trimmed ? `<li>${escapeHtml(trimmed)}</li>` : '';
        }).join('');
        content.innerHTML = `<ul>${items}</ul>`;
      } else {
        content.innerHTML = lines.map(l => `<p>${escapeHtml(l)}</p>`).join('');
      }
    }

    function toggleSummaryPanel() {
      const body    = document.getElementById('summary-body');
      const chevron = document.getElementById('summary-chevron');
      const hidden  = body.style.display === 'none';
      body.style.display = hidden ? '' : 'none';
      chevron.classList.toggle('collapsed', !hidden);
    }

    function resetSummaryPanel() {
      document.getElementById('summary-section').style.display = 'none';
      document.getElementById('summary-content').innerHTML = '';
      document.getElementById('summary-loading').style.display = 'none';
      document.getElementById('summary-error').style.display = 'none';
      document.getElementById('summary-body').style.display = '';
      document.getElementById('summary-chevron').classList.remove('collapsed');
      const btn = document.getElementById('summarize-btn');
      btn.textContent = '✦ Summarize';
      btn.disabled = false;
      if (_summaryEs) { _summaryEs.close(); _summaryEs = null; }
    }

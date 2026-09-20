"use strict";
/* Untangle: local chat UI. No external requests, no dependencies. */
const API = "/api/v1";

/* ============================ pure helpers (unit-tested in node) ============================ */

function escapeHTML(s) {
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function formatInline(escaped) {
  let s = escaped;
  s = s.replace(/\*\*([^\n]+?)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[^*\w])\*([^*\s][^*\n]*?)\*(?![*\w])/g, "$1<em>$2</em>");
  s = s.replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  return s;
}

/* Everything is HTML-escaped BEFORE any markup is added, so model output can never inject tags. */
function renderInline(text) {
  return text.split(/(`[^`\n]+`)/g)
    .map((part, i) => (i % 2 ? `<code>${escapeHTML(part.slice(1, -1))}</code>` : formatInline(escapeHTML(part))))
    .join("");
}

const OL_RE = /^\s*(\d+)[.)]\s+(.*)$/;
const UL_RE = /^\s*[-*+\u2022]\s+(.*)$/;

function renderMarkdown(src) {
  const lines = String(src).replace(/\r\n?/g, "\n").split("\n");
  let html = "", para = [], i = 0;
  const flush = () => {
    if (para.length) { html += `<p>${para.map(renderInline).join("<br>")}</p>`; para = []; }
  };
  while (i < lines.length) {
    const line = lines[i];
    let m;
    if (/^\s*```/.test(line)) {                              // fenced code (may be unclosed while streaming)
      flush();
      const buf = [];
      i++;
      while (i < lines.length && !/^\s*```/.test(lines[i])) buf.push(lines[i++]);
      i++;
      html += `<pre><code>${escapeHTML(buf.join("\n"))}</code></pre>`;
      continue;
    }
    if (!line.trim()) { flush(); i++; continue; }
    if ((m = line.match(/^(#{1,4})\s+(.*)$/))) {
      flush();
      const n = m[1].length + 1;
      html += `<h${n}>${renderInline(m[2])}</h${n}>`;
      i++;
      continue;
    }
    if (/^\s*([-*_])\1{2,}\s*$/.test(line)) { flush(); html += "<hr>"; i++; continue; }
    if (/^>\s?/.test(line)) {
      flush();
      const buf = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) buf.push(lines[i++].replace(/^>\s?/, ""));
      html += `<blockquote>${buf.map(renderInline).join("<br>")}</blockquote>`;
      continue;
    }
    const ordered = OL_RE.test(line);
    if (ordered || UL_RE.test(line)) {
      flush();
      const re = ordered ? OL_RE : UL_RE;
      const start = ordered ? parseInt(line.match(OL_RE)[1], 10) : 1;
      const items = [];
      while (i < lines.length && re.test(lines[i])) {
        let text = lines[i].match(re)[ordered ? 2 : 1];
        i++;
        while (i < lines.length && lines[i].trim() && /^\s{2,}\S/.test(lines[i]) &&
               !OL_RE.test(lines[i]) && !UL_RE.test(lines[i])) {
          text += " " + lines[i].trim();                     // wrapped continuation line
          i++;
        }
        items.push(text);
        let j = i;                                           // allow blank lines between items
        while (j < lines.length && !lines[j].trim()) j++;
        if (j < lines.length && re.test(lines[j])) i = j; else break;
      }
      const lis = items.map((t) => `<li>${renderInline(t)}</li>`).join("");
      html += ordered ? `<ol${start !== 1 ? ` start="${start}"` : ""}>${lis}</ol>` : `<ul>${lis}</ul>`;
      continue;
    }
    para.push(line);
    i++;
  }
  flush();
  return html;
}

/* Incremental Server-Sent-Events parser: feed it the buffer, get complete events + the remainder. */
function parseSSE(buffer) {
  const events = [];
  let rest = buffer, idx;
  while ((idx = rest.indexOf("\n\n")) !== -1) {
    const block = rest.slice(0, idx);
    rest = rest.slice(idx + 2);
    let event = "message";
    const data = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
    }
    if (data.length) {
      try { events.push({ event, data: JSON.parse(data.join("\n")) }); } catch (e) { /* skip bad block */ }
    }
  }
  return { events, rest };
}

function formatMeta(d) {
  const parts = [];
  if (d.ttft_ms != null) parts.push(`first word ${(d.ttft_ms / 1000).toFixed(1)} s`);
  const l = d.llm || {};
  if (l.prompt_tokens != null) parts.push(`read ${l.prompt_tokens} tokens in ${l.prompt_s} s`);
  if (l.gen_tok_per_s) parts.push(`${l.gen_tok_per_s} tokens/s`);
  if (d.prep_ms != null && l.prompt_tokens != null) parts.push(`prep ${(d.prep_ms / 1000).toFixed(1)} s`);
  return parts.join(" \u00b7 ");
}

if (typeof module !== "undefined") module.exports = { escapeHTML, renderMarkdown, parseSSE, formatMeta };

/* ================================== UI (browser only) ================================== */

if (typeof document !== "undefined") {
  const $ = (id) => document.getElementById(id);
  const els = {
    app: $("app"), chat: $("chat"), messages: $("messages"), empty: $("empty"), input: $("input"),
    send: $("sendBtn"), composer: $("composer"), sessionList: $("sessionList"), status: $("status"),
    statusText: $("statusText"), setup: $("setup"), topicList: $("topicList"), setupBtn: $("setupBtn"),
    setupMsg: $("setupMsg"), memoryDialog: $("memoryDialog"), memoryText: $("memoryText"),
  };
  const state = { sid: null, busy: false, controller: null };
  const TYPING = '<span class="typing"><i></i><i></i><i></i></span>';

  async function api(path, opts) {
    const res = await fetch(API + path, opts);
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch (e) { /* not json */ }
      const err = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      err.status = res.status;
      throw err;
    }
    return res.status === 204 ? null : res.json();
  }

  /* ---------- scrolling & rendering ---------- */
  const nearBottom = () => els.chat.scrollHeight - els.chat.scrollTop - els.chat.clientHeight < 140;
  const scrollToBottom = () => { els.chat.scrollTop = els.chat.scrollHeight; };

  function makeRenderer(target, getText) {
    let queued = false;
    const raf = window.requestAnimationFrame || ((f) => setTimeout(f, 16));
    return function render(now) {
      if (now) { target.innerHTML = renderMarkdown(getText()); return; }
      if (queued) return;
      queued = true;
      raf(() => {
        queued = false;
        const stick = nearBottom();
        target.innerHTML = renderMarkdown(getText());
        if (stick) scrollToBottom();
      });
    };
  }

  function createMessage(role) {
    const row = document.createElement("div");
    row.className = `msg ${role}`;
    const body = document.createElement("div");
    body.className = "bubble";
    row.appendChild(body);
    let meta = null;
    if (role === "assistant") {
      meta = document.createElement("div");
      meta.className = "meta";
      row.appendChild(meta);
    }
    els.messages.appendChild(row);
    return { row, body, meta };
  }

  function addUser(text) {
    const m = createMessage("user");
    m.body.textContent = text;
    return m;
  }

  function addCopy(meta, getText) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "mini";
    btn.textContent = "Copy";
    btn.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(getText()); btn.textContent = "Copied"; }
      catch (e) { btn.textContent = "Copy failed"; }
      setTimeout(() => { btn.textContent = "Copy"; }, 1500);
    });
    meta.appendChild(btn);
  }

  function hideEmpty(hide) { els.empty.classList.toggle("hidden", hide); }

  /* ---------- sending ---------- */
  function setBusy(b) {
    state.busy = b;
    els.send.classList.toggle("stop", b);
    els.send.innerHTML = b ? "&#9632;" : "&#8593;";
    els.send.setAttribute("aria-label", b ? "Stop" : "Send");
    updateSendEnabled();
  }
  function updateSendEnabled() { els.send.disabled = !state.busy && !els.input.value.trim(); }

  async function send(text) {
    text = text.trim();
    if (!text || state.busy) return;
    setBusy(true);
    hideEmpty(true);
    addUser(text);
    const a = createMessage("assistant");
    a.body.innerHTML = TYPING;
    scrollToBottom();

    let acc = "", shownFirst = false;
    const render = makeRenderer(a.body, () => acc);
    const controller = new AbortController();
    state.controller = controller;
    const fail = (msg) => {
      a.body.innerHTML = "";
      const s = document.createElement("span");
      s.className = "error-text";
      s.textContent = msg;
      a.body.appendChild(s);
    };

    try {
      const res = await fetch(`${API}/stream_counsel`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, ...(state.sid ? { session_id: state.sid } : {}) }),
        signal: controller.signal,
      });
      if (!res.ok) {
        if (res.status === 409) { fail("Setup isn't finished yet."); await checkSetup(); }
        else if (res.status === 404) { state.sid = null; fail("That conversation no longer exists. Start a new chat."); }
        else if (res.status === 422) fail("That message is empty or longer than 4000 characters.");
        else fail(`The server returned an error (${res.status}).`);
        return;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
        const parsed = parseSSE(buf);
        buf = parsed.rest;
        for (const { event, data } of parsed.events) {
          if (event === "meta") state.sid = data.session_id;
          else if (event === "safety") {
            const n = document.createElement("div");
            n.className = "notice";
            n.textContent = data.message;
            a.row.insertBefore(n, a.body);
          } else if (event === "token") {
            if (!shownFirst) { shownFirst = true; a.body.innerHTML = ""; }
            acc += data.text;
            render();
          } else if (event === "done") {
            render(true);
            a.meta.append(document.createTextNode(formatMeta(data)));
            if (acc) addCopy(a.meta, () => acc);
          } else if (event === "error") {
            if (acc) { render(true); a.meta.append(document.createTextNode(`Error: ${data.message}`)); }
            else fail(data.message);
          }
        }
      }
    } catch (err) {
      if (err.name === "AbortError") {
        if (acc) { render(true); a.meta.append(document.createTextNode("stopped")); }
        else a.row.remove();
      } else fail("Couldn't reach the server. Is it still running?");
    } finally {
      if (acc) render(true);
      state.controller = null;
      setBusy(false);
      els.input.focus();
      scrollToBottom();
      loadSessions();
    }
  }

  /* ---------- sessions ---------- */
  function clearMessages() { els.messages.innerHTML = ""; }

  async function loadSessions() {
    let list = [];
    try { list = await api("/sessions"); } catch (e) { return; }
    els.sessionList.innerHTML = "";
    for (const s of list) {
      const li = document.createElement("li");
      li.className = "session" + (s.id === state.sid ? " active" : "");
      const title = document.createElement("span");
      title.className = "s-title";
      title.textContent = s.title || "New conversation";
      const del = document.createElement("button");
      del.className = "s-del";
      del.type = "button";
      del.title = "Delete conversation";
      del.textContent = "\u00d7";
      del.addEventListener("click", async (ev) => {
        ev.stopPropagation();
        if (!confirm("Delete this conversation? This can't be undone.")) return;
        try { await api(`/sessions/${s.id}`, { method: "DELETE" }); } catch (e) { /* already gone */ }
        if (state.sid === s.id) newChat();
        loadSessions();
      });
      li.append(title, del);
      li.addEventListener("click", () => openSession(s.id));
      els.sessionList.appendChild(li);
    }
  }

  async function openSession(id) {
    if (state.busy) return;
    let d;
    try { d = await api(`/sessions/${id}`); } catch (e) { return; }
    state.sid = id;
    clearMessages();
    hideEmpty(d.messages.length > 0);
    for (const m of d.messages) {
      if (m.role === "user") addUser(m.content);
      else {
        const a = createMessage("assistant");
        a.body.innerHTML = renderMarkdown(m.content);
        addCopy(a.meta, () => m.content);
      }
    }
    els.app.classList.remove("sidebar-open");
    scrollToBottom();
    loadSessions();
  }

  function newChat() {
    if (state.busy) return;
    state.sid = null;
    clearMessages();
    hideEmpty(false);
    els.app.classList.remove("sidebar-open");
    loadSessions();
    els.input.focus();
  }

  /* ---------- status ---------- */
  async function refreshStatus() {
    let text = "Server unreachable", cls = "bad";
    try {
      const h = await api("/health");
      if (h.status === "ok") { text = `Ready \u00b7 ${h.ollama.model}`; cls = "ok"; }
      else if (!h.ollama.reachable) text = "Ollama isn't running";
      else if (!h.ollama.model_available) text = `Model ${h.ollama.model} not installed`;
      else text = "Something needs attention";
    } catch (e) { /* keep defaults */ }
    els.status.className = `status ${cls}`;
    els.statusText.textContent = text;
  }

  /* ---------- first-run setup ---------- */
  function showSetup(show) {
    els.setup.classList.toggle("hidden", !show);
    els.chat.classList.toggle("hidden", show);
    els.composer.parentElement.classList.toggle("hidden", show);
  }

  async function checkSetup() {
    let s;
    try { s = await api("/setup/status"); } catch (e) { return; }
    if (s.initialized && s.state !== "running") { showSetup(false); return; }
    showSetup(true);
    if (!els.topicList.children.length) {
      let topics = [];
      try { topics = await api("/setup/topics"); } catch (e) { /* ignore */ }
      for (const t of topics) {
        const label = document.createElement("label");
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.value = t.id;
        cb.checked = ["cbt", "ifs", "nvc"].includes(t.id);
        label.append(cb, document.createTextNode(t.label));
        els.topicList.appendChild(label);
      }
    }
    if (s.state === "running") pollSetup();
  }

  async function pollSetup() {
    els.setupBtn.disabled = true;
    for (;;) {
      let s;
      try { s = await api("/setup/status"); } catch (e) { els.setupMsg.textContent = "Lost contact with the server."; break; }
      if (s.state === "running") {
        els.setupMsg.textContent =
          `Indexing\u2026 ${s.pages_done + s.pages_failed}/${s.pages_total} pages, ${s.chunks_indexed} passages`;
        await new Promise((r) => setTimeout(r, 1500));
        continue;
      }
      if (s.state === "completed") {
        els.setupMsg.textContent = `Done: ${s.chunks_indexed} passages indexed.`;
        showSetup(false);
        refreshStatus();
      } else els.setupMsg.textContent = `Setup failed: ${s.error || "unknown error"}`;
      break;
    }
    els.setupBtn.disabled = false;
  }

  els.setupBtn.addEventListener("click", async () => {
    const topics = [...els.topicList.querySelectorAll("input:checked")].map((c) => c.value);
    if (!topics.length) { els.setupMsg.textContent = "Pick at least one topic."; return; }
    els.setupMsg.textContent = "Starting\u2026";
    try {
      await api("/setup/warmup", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ topics }) });
      pollSetup();
    } catch (e) { els.setupMsg.textContent = e.message; }
  });

  /* ---------- memory dialog ---------- */
  const openDialog = (d) => (d.showModal ? d.showModal() : d.setAttribute("open", ""));
  const closeDialog = (d) => (d.close ? d.close() : d.removeAttribute("open"));

  $("memoryBtn").addEventListener("click", async () => {
    try { els.memoryText.value = (await api("/profile")).summary; } catch (e) { els.memoryText.value = ""; }
    openDialog(els.memoryDialog);
  });
  $("memoryClose").addEventListener("click", () => closeDialog(els.memoryDialog));
  $("memorySave").addEventListener("click", async () => {
    try {
      await api("/profile", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ summary: els.memoryText.value }) });
      closeDialog(els.memoryDialog);
    } catch (e) { alert(`Couldn't save: ${e.message}`); }
  });
  $("memoryClear").addEventListener("click", async () => {
    if (!confirm("Erase everything I remember about you?")) return;
    try { await api("/profile", { method: "DELETE" }); els.memoryText.value = ""; } catch (e) { alert(`Couldn't erase: ${e.message}`); }
  });

  /* ---------- theme ---------- */
  function applyTheme(t) {
    document.documentElement.dataset.theme = t;
    try { localStorage.setItem("theme", t); } catch (e) { /* storage unavailable */ }
  }
  $("themeBtn").addEventListener("click", () =>
    applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));

  /* ---------- composer wiring ---------- */
  function autosize() {
    els.input.style.height = "auto";
    els.input.style.height = Math.min(els.input.scrollHeight, 200) + "px";
  }
  els.input.addEventListener("input", () => { autosize(); updateSendEnabled(); });
  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); els.composer.requestSubmit(); }
  });
  els.composer.addEventListener("submit", (e) => {
    e.preventDefault();
    if (state.busy) { if (state.controller) state.controller.abort(); return; }
    const text = els.input.value;
    els.input.value = "";
    autosize();
    send(text);
  });
  document.querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => send(c.textContent)));
  $("newChat").addEventListener("click", newChat);
  $("menuBtn").addEventListener("click", () => els.app.classList.toggle("sidebar-open"));

  /* ---------- boot ---------- */
  (async function init() {
    let theme = null;
    try { theme = localStorage.getItem("theme"); } catch (e) { /* ignore */ }
    if (!theme) theme = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    document.documentElement.dataset.theme = theme;
    updateSendEnabled();
    refreshStatus();
    setInterval(refreshStatus, 15000);
    await checkSetup();
    loadSessions();
    els.input.focus();
  })();
}

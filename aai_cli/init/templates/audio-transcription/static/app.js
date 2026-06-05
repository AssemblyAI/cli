const APP_CONFIG = {
  sampleUrl: "https://assembly.ai/wildfires.mp3",
  pollIntervalMs: 2000,
  speakerPalette: ["#171717", "#525252", "#737373", "#262626", "#404040", "#a3a3a3"],
};

const els = {
  url: document.getElementById("url"),
  go: document.getElementById("go"),
  file: document.getElementById("file"),
  status: document.getElementById("status"),
  tabs: document.getElementById("tabs"),
  view: document.getElementById("view"),
  askPanel: document.getElementById("ask"),
  question: document.getElementById("q"),
  ask: document.getElementById("askBtn"),
  answer: document.getElementById("answer"),
};

let currentId = null;
let speakerSeen = {};

els.go.addEventListener("click", () => transcribeUrl(els.url.value.trim()));
els.url.addEventListener("keydown", (event) => {
  if (event.key === "Enter") els.go.click();
});
els.file.addEventListener("change", () => {
  if (els.file.files[0]) transcribeFile(els.file.files[0]);
});
els.ask.addEventListener("click", () => ask(els.question.value.trim()));
els.question.addEventListener("keydown", (event) => {
  if (event.key === "Enter") els.ask.click();
});

async function transcribeUrl(url) {
  if (!url) return;
  await start(() =>
    fetch("/api/transcribe-url", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    })
  );
}

async function transcribeFile(file) {
  const body = new FormData();
  body.append("file", file);
  await start(() => fetch("/api/transcribe", { method: "POST", body }));
}

async function start(submit) {
  busy(true);
  currentId = null;
  speakerSeen = {};
  els.tabs.replaceChildren();
  els.view.replaceChildren();
  els.answer.textContent = "";
  els.askPanel.hidden = true;
  setStatus("Uploading...", "working");

  try {
    const res = await submit();
    if (!res.ok) return fail(await res.text());
    await poll((await res.json()).id);
  } catch (error) {
    fail(error.message || String(error));
  }
}

async function poll(id) {
  setStatus("Transcribing...", "working");
  const res = await fetch("/api/status/" + encodeURIComponent(id));
  if (!res.ok) return fail(await res.text());

  const data = await res.json();
  if (data.status !== "completed") {
    window.setTimeout(() => poll(id).catch((error) => fail(String(error))), APP_CONFIG.pollIntervalMs);
    return;
  }

  setStatus("Done", "done");
  busy(false);
  currentId = id;
  els.askPanel.hidden = false;
  explore(data.transcript);
}

async function ask(question) {
  if (!question || !currentId) return;
  els.ask.disabled = true;
  els.answer.textContent = "Thinking...";
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transcript_id: currentId, question }),
    });
    const data = await res.json();
    els.answer.textContent = res.ok ? data.answer : "Error: " + (data.detail || res.statusText);
  } catch (error) {
    els.answer.textContent = "Error: " + (error.message || String(error));
  } finally {
    els.ask.disabled = false;
  }
}

function speakerColor(speaker) {
  return (speakerSeen[speaker] ??=
    APP_CONFIG.speakerPalette[Object.keys(speakerSeen).length % APP_CONFIG.speakerPalette.length]);
}

function explore(transcript) {
  const views = [
    { label: "Transcript", render: () => renderTranscript(transcript) },
  ];

  if (transcript.chapters?.length) {
    views.push({ label: `Chapters - ${transcript.chapters.length}`, render: () => renderChapters(transcript.chapters) });
  }
  if (transcript.sentiment_analysis_results?.length) {
    views.push({ label: "Sentiment", render: () => renderSentiment(transcript.sentiment_analysis_results) });
  }
  if (transcript.entities?.length) {
    views.push({ label: `Entities - ${transcript.entities.length}`, render: () => renderEntities(transcript.entities) });
  }
  const highlights = transcript.auto_highlights_result?.results || [];
  if (highlights.length) {
    views.push({ label: "Highlights", render: () => renderHighlights(highlights) });
  }

  renderTabs(views);
}

function renderTabs(views) {
  els.tabs.replaceChildren();
  for (const [index, view] of views.entries()) {
    const button = document.createElement("button");
    button.className = "tab-button";
    button.textContent = view.label;
    button.addEventListener("click", () => {
      els.view.replaceChildren(view.render());
      for (const child of els.tabs.children) child.classList.toggle("is-active", child === button);
    });
    els.tabs.appendChild(button);
    if (index === 0) button.click();
  }
}

function renderTranscript(transcript) {
  const turns = transcript.utterances || [];
  if (turns.length) {
    return fragment(turns.map((turn) => turnNode(turn.speaker, turn.text)));
  }
  return element("pre", { className: "transcript-pre" }, transcript.text || "");
}

function renderSentiment(results) {
  return fragment(results.map((item) => {
    const pill = element("span", { className: "sentiment-pill" }, item.sentiment || "");
    if (["POSITIVE", "NEGATIVE", "NEUTRAL"].includes(item.sentiment)) {
      pill.dataset.sentiment = item.sentiment.toLowerCase();
    }
    return turnNode(item.speaker || "?", item.text, pill);
  }));
}

function renderChapters(chapters) {
  return fragment(chapters.map((chapter) => {
    const node = element("article", { className: "chapter-card" });
    node.append(
      element("h4", {}, chapter.headline || chapter.gist || "Chapter"),
      element("span", { className: "timestamp" }, `${fmt(chapter.start)} - ${fmt(chapter.end)}`),
      element("p", { className: "chapter-summary" }, chapter.summary || chapter.gist || "")
    );
    return node;
  }));
}

function renderEntities(entities) {
  const groups = {};
  for (const entity of entities) {
    (groups[entity.entity_type] ??= []).push(entity.text);
  }
  return fragment(Object.entries(groups).map(([type, items]) => {
    const group = element("section", { className: "entity-group" }, element("div", { className: "entity-label" }, type));
    for (const text of new Set(items)) {
      group.appendChild(element("span", { className: "entity-tag" }, text));
    }
    return group;
  }));
}

function renderHighlights(results) {
  return fragment([...results].sort((a, b) => b.rank - a.rank).map((highlight) =>
    element("div", { className: "highlight-row" }, element("span", { className: "highlight-count" }, `${highlight.count}x`), " ", highlight.text)
  ));
}

function turnNode(speaker, text, extra = null) {
  const color = speakerColor(speaker);
  const node = element("article", { className: "transcript-turn" });
  node.style.borderLeftColor = color;

  const name = element("span", { className: "speaker-label" }, `Speaker ${speaker}`);
  name.style.color = color;
  node.append(name);
  if (extra) node.append(extra);
  node.append(element("div", { className: "transcript-text" }, text));
  return node;
}

function fmt(ms) {
  const seconds = Math.round((ms || 0) / 1000);
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

function busy(on) {
  els.go.disabled = on;
}

function setStatus(message, state) {
  els.status.textContent = message;
  els.status.dataset.state = state;
}

function fail(message) {
  busy(false);
  setStatus("Error: " + message, "error");
}

function element(tag, options = {}, ...children) {
  const node = document.createElement(tag);
  if (options.className) node.className = options.className;
  node.append(...children.filter((child) => child !== null && child !== undefined));
  return node;
}

function fragment(children) {
  const node = document.createDocumentFragment();
  node.append(...children);
  return node;
}

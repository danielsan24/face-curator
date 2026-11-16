// Face Curator review page. No inline script and no HTML built from strings: every name, date or other text
// that comes from the server (config names, Synology person names) is set with textContent only.
"use strict";
const $ = s => document.querySelector(s);
const main = $("#main");
// The session token arrives in the URL once. It is kept for this tab only (sessionStorage survives a reload,
// not a closed tab) and removed from the address bar, so it does not stay visible or get copied with the URL.
const TOKEN = (() => {
  let t = new URLSearchParams(location.search).get("t") || "";
  try {
    if (t) sessionStorage.setItem("fc-token", t); else t = sessionStorage.getItem("fc-token") || "";
    if (location.search) history.replaceState(null, "", location.pathname);
  } catch (e) { /* storage blocked: the token stays in the URL */ }
  return t;
})();
let view = "gal", batch = null, deck = [], pos = 0, hist = [], confirmReplace = null;

// el("div", {class: "x", onclick: f}, child, "text", ...): children that are strings become text nodes.
function el(tag, attrs, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (k === "data") Object.assign(n.dataset, v);
    else n.setAttribute(k, v);
  }
  for (const kid of kids.flat(Infinity)) if (kid != null && kid !== false) n.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  return n;
}
const kbd = t => el("kbd", null, t);
const show = (...nodes) => main.replaceChildren(...nodes.flat(Infinity).filter(n => n != null && n !== false));
const img = (path, lazy) => el("img", {src: path, alt: "", loading: lazy ? "lazy" : null});
const crop = (face, size) => `/crop/${encodeURIComponent(face)}?s=${size}`;
const photo = face => `/photo/${encodeURIComponent(face)}?w=900`;

function say(t, bad) { const m = $("#msg"); m.textContent = t; m.className = bad ? "err" : "muted"; }

// Every API call carries the session token as a header. A failed call is shown and thrown: nothing is ever
// treated as saved unless the server said so.
async function api(url, body) {
  const opt = {headers: {"X-FC-Token": TOKEN}, cache: "no-store"};
  if (body !== undefined) { opt.method = "POST"; opt.headers["Content-Type"] = "application/json"; opt.body = JSON.stringify(body); }
  let r, data = null;
  try { r = await fetch(url, opt); } catch (e) {
    say("Not saved: the review server does not answer. Is `facecurator review` still running?", true); throw e;
  }
  try { data = await r.json(); } catch (e) { data = null; }
  if (!r.ok) {
    const err = new Error((data && data.error) || `HTTP ${r.status}`); err.status = r.status; err.data = data;
    if (r.status !== 409) say(`Not saved: ${err.message}`, true);
    throw err;
  }
  return data;
}
const get = u => api(u);
const post = (u, b) => api(u, b);

function tab(v) {
  view = v; confirmReplace = null;
  for (const id of ["gal", "cards", "audit", "cmp"]) $("#t-" + id).classList.toggle("on", id === v);
  const go = v === "gal" ? galleries() : v === "cmp" ? compareView() : cardsView(v === "audit" ? "audit" : "");
  go.catch(() => {});
}
for (const id of ["gal", "cards", "audit", "cmp"]) $("#t-" + id).addEventListener("click", () => tab(id));

async function galleries() {
  batch = null;
  const s = await get("/api/status");
  const w = Object.entries(s.cards_waiting).map(([k, v]) => `${k} ${v}`).join(", ") || "none";
  const out = [el("p", {class: "muted"}, `Target ${s.target} confirmed faces per cell (max ${s.max}); a cell is used from ${s.min_active}. Cards waiting: ${w}.`)];
  for (const p of s.people) {
    const cells = p.bins.map(b => el("td", {
      class: "cell " + (b.n >= s.target ? "ok" : (b.n >= s.min_active ? "thin" : "none")),
      onclick: () => startBatch(p.key, b.bin, p.name, b.label).catch(() => {}),     // names travel as data, never via the DOM
    }, b.n));
    out.push(el("h3", null, p.name), el("table", {class: "cov"}, el("tr", null, p.bins.map(b => el("th", null, b.label))), el("tr", null, cells)));
  }
  out.push(el("p", {class: "muted"}, "Click a cell to review a batch. All tiles start as ", el("b", null, "yes"),
    "; click a tile that is not that person (red), shift-click one that is not a face (amber). ", kbd("Enter"), " saves."));
  show(out);
}

async function startBatch(person, bin, name, label) {
  say("loading…");
  const tiles = await get(`/api/gallery?person=${encodeURIComponent(person)}&bin=${encodeURIComponent(bin)}`);
  say("");
  if (!tiles.length) {
    show(el("p", null, `No unlabelled candidates for ${name} ${label}. `, el("button", {class: "btn alt", onclick: () => galleries().catch(() => {})}, "Back")));
    return;
  }
  batch = {person, tiles: tiles.map(t => ({...t, state: "yes"}))};
  const grid = el("div", {class: "grid"});
  for (const t of batch.tiles) {
    const d = el("div", {class: "tile"}, img(crop(t.face, 240), true), el("small", null, t.day + (t.age != null ? " · " + t.age + "y" : "")));
    d.addEventListener("click", e => {
      t.state = e.shiftKey ? (t.state === "nf" ? "yes" : "nf") : (t.state === "no" ? "yes" : "no");
      d.className = "tile" + (t.state === "no" ? " no" : t.state === "nf" ? " nf" : "");
    });
    grid.append(d);
  }
  show(el("div", {class: "bar"}, el("b", null, `${name} · ${label}`), el("span", {class: "muted"}, `${tiles.length} faces`),
    el("button", {class: "btn", onclick: saveBatch}, "Save (Enter)"),
    el("button", {class: "btn alt", onclick: () => galleries().catch(() => {})}, "Back")), grid);
}

let saving = false;
async function saveBatch() {
  if (!batch || saving) return;
  const b = {person: batch.person, yes: [], no: [], not_face: []};
  for (const t of batch.tiles) (t.state === "yes" ? b.yes : t.state === "no" ? b.no : b.not_face).push(t.face);
  saving = true;
  try {
    const r = await post("/api/gallery", b);          // on an error the batch stays on screen, to save again
    say(`saved ${r.saved}`);
    await galleries();
  } catch (e) { /* shown by api() */ } finally { saving = false; }
}

async function cardsView(kind) {
  deck = await get(`/api/cards?kind=${encodeURIComponent(kind)}`); pos = 0; hist = [];
  renderCard();
}

function side(s, key) {
  return el("div", {class: "side"}, el("h3", null, kbd(key), " ", s.name), el("div", {class: "refs"}, s.refs.map(f => img(crop(f, 160), true))));
}

function renderCard() {
  confirmReplace = null;
  if (pos >= deck.length) {
    show(el("p", null, `No open cards${deck.length ? " left in this deck" : ""}. `, hist.length ? [kbd("Z"), " undoes the last answer."] : null));
    return;
  }
  const c = deck[pos];
  show(el("div", {class: "bar"}, el("b", null, `Card ${pos + 1} of ${deck.length}`), el("span", {class: "muted"}, `${c.kind} · ${c.day} · ${c.bin}`)),
    el("div", {class: "card"},
      el("div", null, el("div", {class: "face"}, img(crop(c.face, 320))), el("p", null, "Who is this?"), side(c.a, "A"), el("br"), side(c.b, "B"),
        el("p", {class: "muted"}, kbd("A"), " / ", kbd("B"), " · ", kbd("N"), " neither · ", kbd("U"), " can't tell · ", kbd("X"), " not a face · ",
          kbd("S"), " skip · ", kbd("Z"), " undo")),
      el("div", {class: "photo"}, img(photo(c.face)))));
}

// An answered card (answered in another tab, say) is not overwritten: the server answers 409 and the same key
// must be pressed a second time, which sends `replace: true` on purpose.
async function send(url, c, a, hist) {
  const again = confirmReplace && confirmReplace.id === c.id && confirmReplace.answer === a;
  try {
    const r = await post(url, again ? {id: c.id, answer: a, replace: true} : {id: c.id, answer: a});
    confirmReplace = null;
    hist.push(c.id);
    return r;
  } catch (e) {
    if (e.status === 409) {
      confirmReplace = {id: c.id, answer: a};
      say("This card already has an answer. Press the same key again to replace it (the earlier answer is kept in the ledger), or S to move on.", true);
    }
    return null;
  }
}

async function answer(a) {
  const c = deck[pos]; if (!c) return;
  if (a === "skip" && confirmReplace && confirmReplace.id === c.id) { confirmReplace = null; say(""); pos++; renderCard(); return; }
  const r = await send("/api/card", c, a, hist);
  if (!r) return;
  say(r.error ? "answer recorded: that was a wrong acceptance; its threshold is raised" : "");
  pos++; renderCard();
}
async function undo() {
  const id = hist.pop(); if (id == null) return;
  try { await post("/api/card/undo", {id}); } catch (e) { hist.push(id); return; }
  pos = Math.max(0, pos - 1); say("undone"); renderCard();
}

// --- blind comparison: the page never receives either system's answer ---
let cmp = null, cpos = 0, chist = [];
const KEYS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "q", "w", "e", "r", "t", "y"];   // o u x s z are reserved
const OKEYS = ["a", "d", "f", "g"];   // per-card names (outer-circle check), in random order
async function compareView() { cmp = await get("/api/compare"); cpos = 0; chist = []; renderCmp(); }
const pick = (a, key, label) => el("button", {class: "btn alt", onclick: () => cmpAnswer(a)}, kbd(key), " ", label);
function renderCmp() {
  confirmReplace = null;
  if (!cmp.cards.length || cpos >= cmp.cards.length) {
    show(el("p", null, `No open comparison cards (${cmp.answered + chist.length} answered). Run `, el("code", null, "facecurator compare report"), " for the result."));
    return;
  }
  const c = cmp.cards[cpos];
  show(el("div", {class: "bar"}, el("b", null, `Who is this? ${cpos + 1} of ${cmp.cards.length}`), el("span", {class: "muted"}, c.day)),
    el("div", {class: "card"},
      el("div", null, el("div", {class: "face"}, img(crop(c.face, 320))),
        el("div", {class: "bar"}, cmp.people.map((p, i) => [pick(p.key, KEYS[i] || "", p.name), " "])),
        c.options && c.options.length ? el("div", {class: "bar"}, c.options.map((p, i) => [pick(p.key, OKEYS[i], p.name), " "])) : null,
        el("div", {class: "bar"}, pick("other", "O", "someone else"), pick("unknown", "U", "can't tell"), pick("not_face", "X", "not a face"), pick("skip", "S", "skip")),
        el("p", {class: "muted"}, "Neither system's answer is shown. ", kbd("Z"), " undoes the last answer.")),
      el("div", {class: "photo"}, img(photo(c.face)))));
}
async function cmpAnswer(a) {
  const c = cmp.cards[cpos]; if (!c) return;
  if (a === "skip" && confirmReplace && confirmReplace.id === c.id) { confirmReplace = null; say(""); cpos++; renderCmp(); return; }
  const r = await send("/api/compare", c, a, chist);
  if (!r) return;
  cpos++; say(""); renderCmp();
}
async function cmpUndo() {
  const id = chist.pop(); if (id == null) return;
  try { await post("/api/compare/undo", {id}); } catch (e) { chist.push(id); return; }
  cpos = Math.max(0, cpos - 1); say("undone"); renderCmp();
}

document.addEventListener("keydown", e => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (view === "gal") { if (e.key === "Enter" && batch) saveBatch(); return; }
  const k = e.key.toLowerCase();
  if (view === "cmp") {
    if (!cmp) return;
    const n = KEYS.indexOf(k), oc = cmp.cards[cpos] && cmp.cards[cpos].options, on = OKEYS.indexOf(k);
    if (n >= 0 && n < cmp.people.length) cmpAnswer(cmp.people[n].key);
    else if (oc && on >= 0 && on < oc.length) cmpAnswer(oc[on].key);
    else if ({o: 1, u: 1, x: 1, s: 1}[k]) cmpAnswer({o: "other", u: "unknown", x: "not_face", s: "skip"}[k]);
    else if (k === "z") cmpUndo();
    return;
  }
  const map = {a: "a", b: "b", n: "neither", u: "unknown", x: "not_face", s: "skip"};
  if (map[k]) answer(map[k]); else if (k === "z") undo();
});

if (!TOKEN) say("Open the URL that `facecurator review` printed: it carries this session's token.", true);
else galleries().catch(() => {});

/* Manndeshi Sahayak — chat client (no framework, works on GitHub Pages). */
(() => {
  "use strict";

  // ------------------------------------------------------------ config & ids
  const params = new URLSearchParams(location.search);
  const API = (params.get("api") || window.SAHAYAK_API || "").replace(/\/$/, "");
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode: in-memory only */ } },
  };
  const rid = () => (crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2) + Date.now().toString(36)).replace(/-/g, "");
  let userId = store.get("sahayak.user") || rid();
  store.set("sahayak.user", userId);
  let sessionId = store.get("sahayak.session") || null;
  let lang = store.get("sahayak.lang") || "auto";

  // ------------------------------------------------------------ ui strings
  const T = {
    mr: { newChat: "नवीन संवाद", history: "मागील संवाद", memory: "माझ्याबद्दल आठवण", tagline: "आर्थिक साक्षरता आणि बँक कर्ज योजनांविषयी विचारा",
      emptyTitle: "नमस्कार! तुम्हाला काय जाणून घ्यायचे आहे?", emptyText: "बचत, बजेट, कर्ज, विमा, बचत गट, डिजिटल पेमेंट आणि माण देशी बँकेच्या उद्योजिका कर्ज योजनेबद्दल प्रश्न विचारा. प्रत्येक उत्तरासोबत स्रोत दिसतील.",
      footnote: "उत्तरे माणदेशी प्रशिक्षण साहित्यावर आधारित आहेत. कर्जाचे अंतिम निर्णय शाखेत विचारून घ्या.", memoryTitle: "मला तुमच्याबद्दल काय आठवते",
      memoryHelp: "संवादातून लक्षात ठेवलेल्या गोष्टी. उत्तरे अधिक उपयुक्त करण्यासाठी वापरल्या जातात. कोणतीही गोष्ट काढून टाकू शकता.", forgetAll: "सर्व विसरा", close: "बंद करा",
      placeholder: "तुमचा प्रश्न इथे लिहा…", sources: "स्रोत", listen: "ऐका", stop: "थांबा", copy: "कॉपी", copied: "कॉपी केले", page: "पान",
      noMemory: "अजून काहीही लक्षात ठेवलेले नाही.", understood: "समजलेला प्रश्न", offline: "सर्व्हर सुरू होत आहे… पहिल्यांदा एक-दोन मिनिटे लागू शकतात.",
      error: "माफ करा, काहीतरी चुकले. पुन्हा प्रयत्न करा.", listening: "ऐकत आहे… पुन्हा दाबा थांबवण्यासाठी", micDenied: "मायक्रोफोन वापरण्याची परवानगी द्या.", confirmForget: "तुमचे सर्व संवाद आणि आठवणी पुसायच्या?",
      remembered: "लक्षात ठेवले:", examples: ["उद्योजिका कर्जासाठी पात्रता काय आहे?", "बचत गट म्हणजे काय आणि त्याचे फायदे?", "महिन्याचे बजेट कसे बनवावे?", "विमा का घ्यावा?", "CGTMSE म्हणजे काय?", "ATM वापरताना काय काळजी घ्यावी?"] },
    hi: { newChat: "नई बातचीत", history: "पिछली बातचीत", memory: "मेरे बारे में याद", tagline: "वित्तीय साक्षरता और बैंक ऋण योजनाओं के बारे में पूछें",
      emptyTitle: "नमस्ते! आप क्या जानना चाहती हैं?", emptyText: "बचत, बजट, ऋण, बीमा, स्वयं सहायता समूह, डिजिटल भुगतान और माण देशी बैंक की उद्योजिका ऋण योजना के बारे में पूछें। हर उत्तर के साथ स्रोत दिखेंगे।",
      footnote: "उत्तर माणदेशी प्रशिक्षण सामग्री पर आधारित हैं। ऋण के अंतिम निर्णय शाखा से पूछें।", memoryTitle: "मुझे आपके बारे में क्या याद है",
      memoryHelp: "बातचीत से याद रखी गई बातें, ताकि उत्तर ज़्यादा उपयोगी हों। आप कोई भी बात हटा सकती हैं।", forgetAll: "सब भूल जाएं", close: "बंद करें",
      placeholder: "अपना सवाल यहाँ लिखें…", sources: "स्रोत", listen: "सुनें", stop: "रोकें", copy: "कॉपी", copied: "कॉपी हुआ", page: "पृष्ठ",
      noMemory: "अभी कुछ याद नहीं रखा गया है।", understood: "समझा गया सवाल", offline: "सर्वर शुरू हो रहा है… पहली बार एक-दो मिनट लग सकते हैं।",
      error: "माफ़ कीजिए, कुछ गड़बड़ हुई। फिर से कोशिश करें।", listening: "सुन रहा है… रोकने के लिए फिर दबाएँ", micDenied: "माइक्रोफ़ोन की अनुमति दें।", confirmForget: "आपकी सारी बातचीत और यादें मिटा दें?",
      remembered: "याद रखा:", examples: ["उद्योजिका ऋण के लिए पात्रता क्या है?", "स्वयं सहायता समूह क्या है?", "महीने का बजट कैसे बनाएँ?", "बीमा क्यों लेना चाहिए?"] },
    en: { newChat: "New chat", history: "Past chats", memory: "What you remember", tagline: "Ask about financial literacy and bank loan schemes",
      emptyTitle: "Hello! What would you like to know?", emptyText: "Ask about savings, budgeting, loans, insurance, self-help groups, digital payments and Mann Deshi Bank's Udyojika loan. Every answer shows its sources.",
      footnote: "Answers come from Mann Deshi training material. Confirm loan decisions at your branch.", memoryTitle: "What I remember about you",
      memoryHelp: "Things remembered from your chats, used to make answers more relevant. Remove anything you like.", forgetAll: "Forget everything", close: "Close",
      placeholder: "Type your question…", sources: "Sources", listen: "Listen", stop: "Stop", copy: "Copy", copied: "Copied", page: "page",
      noMemory: "Nothing remembered yet.", understood: "Understood as", offline: "The server is waking up… the first request can take a minute or two.",
      error: "Sorry, something went wrong. Please try again.", listening: "Listening… tap again to stop", micDenied: "Please allow microphone access.", confirmForget: "Erase all your chats and memories?",
      remembered: "Remembered:", examples: ["Who is eligible for the Udyojika loan?", "What is a self-help group?", "How do I make a monthly budget?", "Why should I buy insurance?", "What is CGTMSE?"] },
  };
  const ui = () => T[lang === "auto" ? "mr" : lang];

  // ------------------------------------------------------------ dom
  const $ = (id) => document.getElementById(id);
  const chat = $("chat"), input = $("input"), form = $("composer"), sendBtn = $("send"), micBtn = $("mic");
  const banner = $("banner"), langSel = $("lang"), empty = $("empty");

  function applyUi() {
    const t = ui();
    document.documentElement.lang = lang === "auto" ? "mr" : lang;
    document.querySelectorAll("[data-i18n]").forEach((el) => { if (t[el.dataset.i18n]) el.textContent = t[el.dataset.i18n]; });
    input.placeholder = t.placeholder;
    const ex = $("examples");
    ex.replaceChildren(...t.examples.map((q) => {
      const b = document.createElement("button"); b.className = "chip"; b.type = "button"; b.textContent = q;
      b.onclick = () => send(q); return b;
    }));
  }

  // ------------------------------------------------------------ api
  async function api(path, opts = {}) {
    const res = await fetch(API + path, { ...opts, headers: { "x-user-id": userId, ...(opts.body && !(opts.body instanceof FormData) ? { "Content-Type": "application/json" } : {}), ...(opts.headers || {}) } });
    if (!res.ok) {
      let msg = res.statusText;
      try { msg = (await res.json()).detail || msg; } catch { /* not json */ }
      const err = new Error(msg); err.status = res.status; throw err;
    }
    return res;
  }

  async function checkHealth() {
    try {
      const h = await (await api("/api/health")).json();
      banner.hidden = !!h.ok;
      if (!h.ok) { banner.textContent = ui().offline; setTimeout(checkHealth, 8000); }
    } catch {
      banner.hidden = false; banner.textContent = ui().offline; setTimeout(checkHealth, 8000);
    }
  }

  // ------------------------------------------------------------ rendering
  function esc(s) { return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

  function renderMarkdown(text) {
    // minimal, safe: escape first, then add bold, lists, paragraphs and citation buttons
    const lines = esc(text).split(/\n/);
    let html = "", list = null;
    const inline = (s) => s.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
      .replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (_, ns) => ns.split(/\s*,\s*/).map((n) => `<button class="cite" data-n="${n}" type="button">${n}</button>`).join(""));
    for (const raw of lines) {
      const line = raw.trim();
      const m = line.match(/^([-*•]|\d+[.)])\s+(.*)$/);
      if (m) {
        const tag = /^\d/.test(m[1]) ? "ol" : "ul";
        if (list !== tag) { if (list) html += `</${list}>`; html += `<${tag}>`; list = tag; }
        html += `<li>${inline(m[2])}</li>`;
        continue;
      }
      if (list) { html += `</${list}>`; list = null; }
      if (line.startsWith("#")) html += `<p><b>${inline(line.replace(/^#+\s*/, ""))}</b></p>`;
      else if (line) html += `<p>${inline(line)}</p>`;
    }
    if (list) html += `</${list}>`;
    return html;
  }

  function addUser(text) {
    empty.hidden = true;
    const m = document.createElement("div"); m.className = "msg user";
    const b = document.createElement("div"); b.className = "bubble"; b.textContent = text;
    m.append(b); chat.append(m); scroll();
  }

  function addTyping() {
    const m = document.createElement("div"); m.className = "msg bot";
    m.innerHTML = '<div class="bubble typing" aria-label="…"><span></span><span></span><span></span></div>';
    chat.append(m); scroll(); return m;
  }

  function addBot(data, replace) {
    const t = ui();
    const m = replace || document.createElement("div"); m.className = "msg bot"; m.replaceChildren();
    const b = document.createElement("div"); b.className = "bubble"; b.innerHTML = renderMarkdown(data.answer || "");
    m.append(b);

    const tools = document.createElement("div"); tools.className = "tools";
    const listen = mkTool("🔊 " + t.listen, () => speak(data.answer, data.language, listen));
    const copy = mkTool("📋 " + t.copy, async () => { try { await navigator.clipboard.writeText(data.answer); copy.textContent = "✓ " + t.copied; } catch { /* ignore */ } });
    tools.append(listen, copy);
    let srcBox;
    if (data.sources && data.sources.length) {
      const toggle = mkTool(`📄 ${t.sources} (${data.sources.length})`, () => { srcBox.hidden = !srcBox.hidden; toggle.setAttribute("aria-pressed", String(!srcBox.hidden)); });
      toggle.setAttribute("aria-pressed", "false");
      tools.append(toggle);
      srcBox = document.createElement("div"); srcBox.className = "sources"; srcBox.hidden = true;
      for (const s of data.sources) {
        const d = document.createElement("div"); d.className = "source"; d.id = `src-${m.dataset.k = m.dataset.k || rid()}-${s.n}`;
        d.innerHTML = `<b>[${s.n}]</b> ${esc(s.filename.replace(/\.pdf$/i, ""))} <span class="meta">· ${esc(s.module || "")} · ${t.page} ${s.page}</span>` +
          (s.heading ? `<div class="meta">${esc(s.heading)}</div>` : "") + `<div class="preview">${esc(s.preview || "")}</div>`;
        srcBox.append(d);
      }
    }
    m.append(tools);
    if (srcBox) m.append(srcBox);
    if (data.rewritten && data.message && data.rewritten.trim() !== data.message.trim()) {
      const r = document.createElement("div"); r.className = "rewritten"; r.textContent = `${t.understood}: ${data.rewritten}`; m.append(r);
    }
    if (data.new_facts && data.new_facts.length) {
      const r = document.createElement("div"); r.className = "rewritten"; r.textContent = `🧠 ${t.remembered} ${data.new_facts.join("; ")}`; m.append(r);
    }
    b.querySelectorAll(".cite").forEach((c) => c.onclick = () => {
      if (!srcBox) return; srcBox.hidden = false;
      const el = srcBox.querySelector(`[id$="-${c.dataset.n}"]`);
      if (el) { el.scrollIntoView({ block: "nearest", behavior: "smooth" }); el.classList.add("flash"); setTimeout(() => el.classList.remove("flash"), 1200); }
    });
    if (!replace) chat.append(m);
    scroll();
  }

  function mkTool(label, onclick) { const b = document.createElement("button"); b.type = "button"; b.className = "tool"; b.textContent = label; b.onclick = onclick; return b; }
  function scroll() { chat.scrollTop = chat.scrollHeight; }

  // ------------------------------------------------------------ chat
  let busy = false;
  async function send(text) {
    text = (text || input.value).trim();
    if (!text || busy) return;
    busy = true; sendBtn.disabled = true; input.value = ""; autosize();
    addUser(text);
    const pending = addTyping();
    try {
      const res = await api("/api/chat", { method: "POST", body: JSON.stringify({ user_id: userId, session_id: sessionId, message: text, language: lang }) });
      const data = await res.json();
      data.message = text;
      sessionId = data.session_id; store.set("sahayak.session", sessionId);
      addBot(data, pending);
      loadSessions();
    } catch (e) {
      addBot({ answer: e.status === 429 ? e.message : (e.status === 503 ? ui().offline : ui().error), sources: [] }, pending);
      if (e.status === 503 || !e.status) checkHealth();
    } finally {
      busy = false; sendBtn.disabled = false; input.focus();
    }
  }

  form.addEventListener("submit", (e) => { e.preventDefault(); send(); });
  input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
  function autosize() { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 160) + "px"; }
  input.addEventListener("input", autosize);

  // ------------------------------------------------------------ sessions
  async function loadSessions() {
    try {
      const { sessions } = await (await api(`/api/sessions?user_id=${userId}`)).json();
      const ul = $("sessions");
      ul.replaceChildren(...sessions.map((s) => {
        const li = document.createElement("li"); const b = document.createElement("button");
        b.textContent = s.title || "…"; b.title = s.title || ""; if (s.id === sessionId) b.setAttribute("aria-current", "true");
        b.onclick = () => openSession(s.id); li.append(b); return li;
      }));
    } catch { /* backend asleep */ }
  }

  async function openSession(id) {
    sessionId = id; store.set("sahayak.session", id);
    chat.replaceChildren(empty); empty.hidden = true;
    $("sidebar").classList.remove("open");
    try {
      const { messages } = await (await api(`/api/sessions/${id}?user_id=${userId}`)).json();
      let lastUser = "";
      for (const m of messages) {
        if (m.role === "user") { addUser(m.content); lastUser = m.content; }
        else addBot({ answer: m.content, sources: m.sources, language: m.language, message: lastUser });
      }
      if (!messages.length) empty.hidden = false;
    } catch { empty.hidden = false; }
    loadSessions();
  }

  $("newChat").onclick = () => {
    sessionId = null; store.set("sahayak.session", "");
    chat.replaceChildren(empty); empty.hidden = false; $("sidebar").classList.remove("open"); input.focus(); loadSessions();
  };
  $("toggleSidebar").onclick = () => $("sidebar").classList.toggle("open");

  // ------------------------------------------------------------ memory
  async function showMemory() {
    const dlg = $("memoryDialog"), ul = $("facts");
    ul.replaceChildren();
    try {
      const { facts } = await (await api(`/api/memory?user_id=${userId}`)).json();
      if (!facts.length) { const li = document.createElement("li"); li.textContent = ui().noMemory; ul.append(li); }
      for (const f of facts) {
        const li = document.createElement("li"); const s = document.createElement("span"); s.textContent = f.fact;
        const x = document.createElement("button"); x.type = "button"; x.className = "x"; x.textContent = "✕"; x.ariaLabel = "remove";
        x.onclick = async () => { await api(`/api/memory/${f.id}?user_id=${userId}`, { method: "DELETE" }); li.remove(); };
        li.append(s, x); ul.append(li);
      }
    } catch { const li = document.createElement("li"); li.textContent = ui().offline; ul.append(li); }
    dlg.showModal();
  }
  $("openMemory").onclick = showMemory;
  $("forgetAll").onclick = async () => {
    if (!confirm(ui().confirmForget)) return;
    await api(`/api/memory?user_id=${userId}`, { method: "DELETE" });
    userId = rid(); store.set("sahayak.user", userId); sessionId = null; store.set("sahayak.session", "");
    $("memoryDialog").close(); chat.replaceChildren(empty); empty.hidden = false; loadSessions();
  };

  // ------------------------------------------------------------ voice in
  let rec = null, chunks = [];
  micBtn.onclick = async () => {
    if (rec && rec.state === "recording") { rec.stop(); return; }
    if (!navigator.mediaDevices || !window.MediaRecorder) { alert(ui().micDenied); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const type = ["audio/webm", "audio/ogg", "audio/mp4"].find((t) => MediaRecorder.isTypeSupported(t)) || "";
      rec = new MediaRecorder(stream, type ? { mimeType: type } : undefined); chunks = [];
      rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
      rec.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop()); micBtn.classList.remove("recording"); input.placeholder = ui().placeholder;
        const blob = new Blob(chunks, { type: rec.mimeType || "audio/webm" });
        const ext = (rec.mimeType || "audio/webm").includes("ogg") ? "ogg" : (rec.mimeType || "").includes("mp4") ? "m4a" : "webm";
        const fd = new FormData(); fd.append("file", blob, `speech.${ext}`); fd.append("language", lang);
        micBtn.disabled = true;
        try {
          const r = await (await api("/api/stt", { method: "POST", body: fd })).json();
          if (r.transcript) send(r.transcript);
        } catch { alert(ui().error); }
        finally { micBtn.disabled = false; }
      };
      rec.start(); micBtn.classList.add("recording"); input.placeholder = ui().listening;
      setTimeout(() => rec && rec.state === "recording" && rec.stop(), 28000);   // Sarvam REST STT limit ~30 s
    } catch { alert(ui().micDenied); }
  };

  // ------------------------------------------------------------ voice out
  let audio = null;
  async function speak(text, language, btn) {
    const t = ui();
    if (audio && !audio.paused) { audio.pause(); audio = null; btn.textContent = "🔊 " + t.listen; return; }
    btn.disabled = true;
    try {
      const res = await api("/api/tts", { method: "POST", body: JSON.stringify({ text, language: language || (lang === "auto" ? "mr" : lang) }) });
      audio = new Audio(URL.createObjectURL(await res.blob()));
      audio.onended = () => { btn.textContent = "🔊 " + t.listen; };
      await audio.play(); btn.textContent = "⏹ " + t.stop;
    } catch { alert(t.error); }
    finally { btn.disabled = false; }
  }

  // ------------------------------------------------------------ start
  langSel.value = lang;
  langSel.onchange = () => { lang = langSel.value; store.set("sahayak.lang", lang); applyUi(); };
  applyUi();
  checkHealth();
  if (sessionId) openSession(sessionId); else loadSessions();
})();

/* Μηνυσάκιας — the conversation page.
 *
 * The conversation lives here, in the browser (sessionStorage): each question sends it whole to POST /api/ask and
 * reads the answer back as a stream of JSON lines (status → sources → text … → done | error).
 */
(function () {
  "use strict";

  const STORAGE_KEY = "minisakias.thread.v1";
  const MAX_CHARS = 4000;
  const STATUS = {
    routing: "Ψάχνω στους κώδικες τις διατάξεις που σας αφορούν…",
    writing: "Διαβάζω τα άρθρα και γράφω την απάντηση…",
  };

  const $ = (id) => document.getElementById(id);
  const thread = $("thread"), form = $("composer"), input = $("question"), send = $("send"), hint = $("hint");
  const dialog = $("article-dialog");
  const hintDefault = hint.textContent;

  /** @type {{role: string, content: string, sources?: object[], cited?: object[], error?: string, stopped?: boolean}[]} */
  let messages = [];
  let controller = null;          // the request in flight, if any

  // ---- storage ------------------------------------------------------------------------------------------------
  function save() {
    try { sessionStorage.setItem(STORAGE_KEY, JSON.stringify(messages)); } catch (e) { /* private mode: fine */ }
  }
  function load() {
    try {
      const stored = JSON.parse(sessionStorage.getItem(STORAGE_KEY) || "[]");
      return Array.isArray(stored) ? stored.filter((m) => m && typeof m.content === "string") : [];
    } catch (e) { return []; }
  }

  // ---- citations ----------------------------------------------------------------------------------------------
  const citeKey = (label) => label.toUpperCase().replace(/[\s.]/g, "");
  const BRACKET = /\[([^\[\]\n]{2,80})\]/g;
  const PIECE = /^\s*(?:και\s+)?([A-Za-zΑ-Ωα-ωάέήίόύώ.]{1,12})\s+(\d+\s*[Α-Ωα-ω]{0,2})(?![\dΑ-Ωα-ω])/;

  /** Turn «[ΠΚ 308]» in rendered text into buttons, for the articles this answer is known to have. */
  function linkCitations(root, known) {
    if (!known.size) return;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) {
      const node = walker.currentNode;
      if (node.nodeValue.includes("[") && !node.parentElement.closest("a, code, button")) nodes.push(node);
    }
    for (const node of nodes) {
      const text = node.nodeValue;
      const out = document.createDocumentFragment();
      let last = 0, changed = false, m;
      BRACKET.lastIndex = 0;
      while ((m = BRACKET.exec(text))) {
        const items = m[1].split(/[,;·]/).map((piece) => {
          const p = PIECE.exec(piece);
          return p ? known.get(citeKey(p[1] + p[2])) : null;
        });
        if (!items.length || items.some((x) => !x)) continue;
        out.append(text.slice(last, m.index));
        items.forEach((item, i) => {
          if (i) out.append(" ");
          const b = document.createElement("button");
          b.type = "button";
          b.className = "cite";
          b.dataset.article = item.id;
          b.title = item.heading ? `${item.code_name || item.code} — ${item.heading}` : (item.code_name || item.label);
          b.textContent = item.label;
          out.append(b);
        });
        last = m.index + m[0].length;
        changed = true;
      }
      if (changed) {
        out.append(text.slice(last));
        node.replaceWith(out);
      }
    }
  }

  function knownArticles(message) {
    const known = new Map();
    for (const item of [...(message.sources || []), ...(message.cited || [])]) known.set(citeKey(item.label), item);
    return known;
  }

  // ---- rendering ----------------------------------------------------------------------------------------------
  function renderMarkdown(target, message) {
    const html = window.marked.parse(message.content, { gfm: true, breaks: false });
    target.innerHTML = window.DOMPurify.sanitize(html, {
      FORBID_TAGS: ["style", "img", "form", "input", "iframe", "svg", "math"],
      FORBID_ATTR: ["style", "class", "id"],
    });
    for (const a of target.querySelectorAll("a[href]")) {
      a.target = "_blank";
      a.rel = "noopener noreferrer";
    }
    linkCitations(target, knownArticles(message));
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function renderExtras(box, message) {
    box.replaceChildren();
    const cited = message.cited || [];
    if (cited.length) {
      box.append(el("p", "sources-title", "Διατάξεις στις οποίες στηρίζεται η απάντηση"));
      const list = el("ul", "sources");
      for (const item of cited) {
        const li = el("li"), b = el("button");
        b.type = "button";
        b.dataset.article = item.id;
        b.append(el("b", "", item.label));
        if (item.heading) b.append(" " + item.heading);
        li.append(b);
        list.append(li);
      }
      box.append(list);
    }
    const actions = el("div", "actions");
    const copy = el("button", "", "Αντιγραφή απάντησης");
    copy.type = "button";
    copy.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(message.content);
        copy.textContent = "Αντιγράφηκε";
      } catch (e) { copy.textContent = "Η αντιγραφή δεν επιτράπηκε"; }
      setTimeout(() => { copy.textContent = "Αντιγραφή απάντησης"; }, 2000);
    });
    actions.append(copy);
    box.append(actions);
    box.hidden = false;
  }

  function renderError(node, message, index) {
    node.querySelector(".error")?.remove();
    if (!message.error) return;
    const box = el("div", "error");
    box.setAttribute("role", "alert");
    box.append(el("p", "", message.error));
    if (index === messages.length - 1) {
      const retry = el("button", "", "Δοκιμάστε ξανά");
      retry.type = "button";
      retry.addEventListener("click", () => {
        messages.pop();                       // the failed answer; the question before it is asked again
        ask(null);
      });
      box.append(retry);
    }
    node.append(box);
  }

  /** Build the DOM for one message; returns the handles the stream updates. */
  function mount(message, index) {
    if (message.role === "user") {
      thread.append(el("div", "msg user", message.content));
      return null;
    }
    const node = el("article", "msg assistant");
    const status = el("p", "status");
    status.hidden = true;
    const answer = el("div", "answer");
    const extras = el("div", "extras");
    extras.hidden = true;
    node.append(status, answer, extras);
    thread.append(node);
    if (message.content) renderMarkdown(answer, message);
    if (message.content && !message.error) renderExtras(extras, message);
    renderError(node, message, index);
    return { node, status, answer, extras };
  }

  function renderAll() {
    thread.replaceChildren();
    messages.forEach(mount);
    const chatting = messages.length > 0;
    document.body.classList.toggle("chatting", chatting);
    thread.hidden = !chatting;
    $("new-case").hidden = !chatting;
    input.rows = chatting ? 1 : 3;
    autosize();
    input.placeholder = chatting
      ? "Γράψτε μια διευκρίνιση ή μια νέα ερώτηση…"
      : "Π.χ. «Ο εργοδότης μου δεν με έχει πληρώσει δύο μήνες και απειλεί να με απολύσει αν παραπονεθώ…»";
  }

  function nearBottom() {
    return window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 160;
  }
  function scrollToEnd() {
    window.scrollTo({ top: document.documentElement.scrollHeight });
  }

  // ---- asking -------------------------------------------------------------------------------------------------
  function setBusy(busy) {
    send.textContent = busy ? "Διακοπή" : "Ρωτήστε";
    send.classList.toggle("stop", busy);
    send.type = busy ? "button" : "submit";
    input.disabled = busy;
    $("new-case").disabled = busy;
  }

  /** Ask with `text` as a new message, or (text === null) ask the conversation's last question again. */
  async function ask(text) {
    if (controller) return;
    if (text !== null) messages.push({ role: "user", content: text });
    const reply = { role: "assistant", content: "" };
    messages.push(reply);
    renderAll();
    const ui = [...thread.querySelectorAll(".msg.assistant")].pop();
    const status = ui.querySelector(".status"), answer = ui.querySelector(".answer"), extras = ui.querySelector(".extras");
    status.textContent = STATUS.routing;
    status.hidden = false;
    scrollToEnd();
    setBusy(true);
    controller = new AbortController();

    let pending = false;
    const paint = () => {
      pending = false;
      const stick = nearBottom();
      renderMarkdown(answer, reply);
      if (stick) scrollToEnd();
    };

    try {
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: messages.slice(0, -1).filter((m) => m.content).map((m) => ({ role: m.role, content: m.content })) }),
        signal: controller.signal,
      });
      if (!response.ok) {
        let detail = "";
        try { detail = (await response.json()).detail; } catch (e) { /* not JSON */ }
        throw new Error(typeof detail === "string" && detail ? detail : "Η υπηρεσία δεν είναι διαθέσιμη αυτή τη στιγμή. Δοκιμάστε ξανά σε λίγο.");
      }
      const reader = response.body.getReader(), decoder = new TextDecoder();
      let buffer = "", finished = false;
      const handle = (event) => {
        if (event.type === "status") {
          if (!reply.content) status.textContent = STATUS[event.stage] || status.textContent;
        } else if (event.type === "sources") {
          reply.sources = event.items;
        } else if (event.type === "text") {
          reply.content += event.text;
          status.hidden = true;
          if (!pending) { pending = true; requestAnimationFrame(paint); }
        } else if (event.type === "done") {
          reply.cited = event.cited;
          finished = true;
        } else if (event.type === "error") {
          reply.error = event.message;
          finished = true;
        }
      };
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let nl;
        while ((nl = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, nl).trim();
          buffer = buffer.slice(nl + 1);
          if (line) handle(JSON.parse(line));
        }
      }
      if (!finished) reply.error = "Η σύνδεση διακόπηκε πριν ολοκληρωθεί η απάντηση.";
    } catch (e) {
      if (e.name === "AbortError") reply.stopped = true;
      else reply.error = e.message || "Κάτι πήγε στραβά. Δοκιμάστε ξανά.";
    } finally {
      controller = null;
      setBusy(false);
    }

    status.hidden = true;
    if (reply.stopped && !reply.content) {
      // stopped before a word was written: put the question back in the box
      messages.pop();
      const question = messages.pop();
      renderAll();
      input.value = question ? question.content : "";
      autosize();
    } else {
      const index = messages.length - 1;
      if (reply.content) renderMarkdown(answer, reply);
      if (reply.content && !reply.error) renderExtras(extras, reply);
      renderError(ui, reply, index);
    }
    save();
    input.focus();
  }

  // ---- article dialog -----------------------------------------------------------------------------------------
  async function openArticle(id) {
    $("article-code").textContent = "";
    $("article-title").textContent = "Φόρτωση…";
    $("article-version").hidden = true;
    $("article-text").replaceChildren();
    $("article-source").replaceChildren();
    if (!dialog.open) dialog.showModal();
    try {
      const response = await fetch("/api/articles/" + encodeURIComponent(id));
      if (!response.ok) throw new Error();
      const a = await response.json();
      $("article-code").textContent = a.code_name;
      $("article-title").textContent = "Άρθρο " + a.number + (a.heading ? " — " + a.heading : "");
      $("article-version").textContent = a.version;
      $("article-version").hidden = false;
      const text = $("article-text");
      if (a.chapter) text.append(el("span", "chapter", a.chapter));
      text.append(a.text);
      const source = $("article-source");
      source.append((a.act ? a.act + " · " : "") + "Κείμενο από το αποθετήριο lexifina/greece-law. ");
      if (/^https:\/\//.test(a.source || "")) {
        const link = el("a", "", "Επίσημη δημοσίευση (ΦΕΚ)");
        link.href = a.source;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        source.append(link);
      }
    } catch (e) {
      $("article-title").textContent = "Το άρθρο δεν βρέθηκε";
    }
  }

  // ---- wiring -------------------------------------------------------------------------------------------------
  function autosize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, window.innerHeight * 0.4) + "px";
    const left = MAX_CHARS - input.value.length;
    hint.textContent = left < 400 ? `Απομένουν ${left} χαρακτήρες` : hintDefault;
    hint.classList.toggle("over", left < 50);
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const text = input.value.trim();
    if (!text || controller) return;
    input.value = "";
    autosize();
    ask(text);
  });
  send.addEventListener("click", () => { if (controller) controller.abort(); });
  input.addEventListener("input", autosize);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      form.requestSubmit();
    }
  });
  $("examples").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-text]");
    if (!button) return;
    input.value = button.dataset.text;
    autosize();
    input.focus();
    input.scrollIntoView({ block: "center", behavior: "smooth" });
  });
  $("new-case").addEventListener("click", () => {
    if (controller) return;
    messages = [];
    save();
    renderAll();
    input.value = "";
    autosize();
    window.scrollTo({ top: 0 });
    input.focus();
  });
  document.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-article]");
    if (button) openArticle(button.dataset.article);
  });
  $("article-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); });

  fetch("/health").then((r) => r.json()).then((h) => { $("demo-badge").hidden = !h.demo; }).catch(() => {});

  messages = load();
  // an answer that was still being written when the page closed is kept only if it has text
  if (messages.length && messages[messages.length - 1].role === "assistant" && !messages[messages.length - 1].content) messages.pop();
  renderAll();
  autosize();
})();

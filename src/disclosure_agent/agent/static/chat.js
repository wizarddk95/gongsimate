const ACTIVITY = {
  resolving_company: "기업을 확인하고 있습니다",
  searching_disclosures: "공시 원문을 찾고 있습니다",
  generating_answer: "답변을 작성하고 있습니다",
};

const RETRY_CODES = new Set([
  "request_timeout",
  "retrieval_failure",
  "generation_failure",
  "openai_timeout",
  "openai_api_error",
  "openai_rate_limit",
  "search_limit_exceeded",
  "llm_limit_exceeded",
  "internal_error",
]);

const INLINE_NOTES = new Set([
  "partial",
  "unavailable",
  "missing",
  "abstained",
  "limited",
  "empty",
  "timeout",
  "error",
]);

const PROMPT_TEMPLATES = [
  (name) => `${name}의 주요 사업은 무엇인가?`,
  (name) => `${name}의 최근 실적 변동 이유를 설명해 주세요.`,
  (name) => `${name}의 사업 구조를 공시 원문과 함께 설명해 주세요.`,
];

const transcript = document.querySelector("#transcript");
const emptyState = document.querySelector("#empty");
const form = document.querySelector("#ask-form");
const questionInput = document.querySelector("#question");
const sendButton = document.querySelector("#send");
const activity = document.querySelector("#activity");
const corpusDescription = document.querySelector("#corpus-description");
const corpusList = document.querySelector("#corpus-list");
const coverageEmpty = document.querySelector("#coverage-empty");
const coverageFilter = document.querySelector("#coverage-filter");
const coverageLabel = document.querySelector("#coverage-label");
const coverageToggle = document.querySelector("#coverage-toggle");
const coveragePanel = document.querySelector("#coverage-panel");
const promptList = document.querySelector("#prompt-list");
const drawer = document.querySelector("#drawer");
const scrim = document.querySelector("#scrim");
const drawerMeta = document.querySelector("#drawer-meta");
const drawerText = document.querySelector("#drawer-text");
const drawerTechnical = document.querySelector("#drawer-technical");
const drawerMore = document.querySelector("#drawer-more");
const drawerTitle = document.querySelector("#drawer-title");
const drawerKicker = document.querySelector("#drawer-kicker");
const drawerDocument = document.querySelector("#drawer-document");
const historyList = document.querySelector("#history-list");
const historyOpen = document.querySelector("#history-open");
const historyDismiss = document.querySelector("#history-dismiss");
const homeLink = document.querySelector("#home");
const store = window.GongsiMateStore;

let busy = false;
let drawerTrigger = null;
let coverageTrigger = null;
let corpusCompanies = [];
let session = null;
let activeAssistant = null;
let abortController = null;
let leaving = false;

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = questionInput.value.trim();
  if (!message || busy) return;
  questionInput.value = "";
  resizeQuestion();
  updateSendState();
  ask({ message, display: message });
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

questionInput.addEventListener("input", () => {
  resizeQuestion();
  updateSendState();
});

coverageToggle.addEventListener("click", () => {
  if (coveragePanel.hidden) openCoverage(coverageToggle);
  else closeCoverage();
});
document.querySelector("#coverage-close").addEventListener("click", closeCoverage);
document.querySelector("#coverage-dismiss").addEventListener("click", closeCoverage);
coverageFilter.addEventListener("input", () => renderCompanyList(coverageFilter.value));

document.querySelector("#drawer-close").addEventListener("click", closeDrawer);
scrim.addEventListener("click", closeDrawer);
homeLink.addEventListener("click", (event) => {
  event.preventDefault();
  startNewConversation();
});
document.querySelector("#new-conversation").addEventListener("click", startNewConversation);
historyOpen.addEventListener("click", () => {
  if (document.body.dataset.history === "open") closeHistory();
  else openHistory();
});
historyDismiss.addEventListener("click", closeHistory);
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  if (!drawer.hidden) closeDrawer();
  else if (!coveragePanel.hidden) closeCoverage();
  else if (document.body.dataset.history === "open") closeHistory();
});

loadCorpus();
renderHistory();
resizeQuestion();
updateSendState();

async function loadCorpus() {
  try {
    const response = await fetch("/api/corpus");
    if (!response.ok) throw new Error("corpus");
    renderCorpus(await response.json());
  } catch {
    corpusCompanies = [];
    coverageLabel.textContent = "지원 범위";
    corpusDescription.textContent = "지원 기업 목록을 불러오지 못했습니다.";
    corpusList.replaceChildren();
    coverageEmpty.hidden = true;
    promptList.replaceChildren();
  }
}

function renderCorpus(payload) {
  corpusCompanies = Array.isArray(payload.companies) ? payload.companies : [];
  const count = corpusCompanies.length;
  coverageLabel.textContent = count ? `지원 기업 ${count}곳` : "지원 기업 없음";
  corpusDescription.textContent = payload.description || "지원하는 기업의 공시 원문만 검색합니다.";
  renderCompanyList(coverageFilter.value);
  renderPrompts(corpusCompanies);
}

function renderCompanyList(query) {
  const needle = String(query || "").trim().toLowerCase();
  corpusList.replaceChildren();
  const visible = corpusCompanies.filter((company) => {
    const name = String(company.company_name || "");
    return !needle || name.toLowerCase().includes(needle);
  });
  if (!corpusCompanies.length) {
    const item = document.createElement("li");
    item.textContent = "현재 지원하는 기업이 없습니다.";
    corpusList.append(item);
    coverageEmpty.hidden = true;
    return;
  }
  coverageEmpty.hidden = visible.length > 0;
  for (const company of visible) {
    const item = document.createElement("li");
    const name = document.createElement("p");
    name.className = "company-name";
    name.textContent = company.company_name || "이름 없는 기업";
    item.append(name);
    const meta = document.createElement("p");
    meta.className = "company-meta";
    const bits = [];
    if (company.corpus_as_of) bits.push(`데이터 기준 ${formatAsOf(company.corpus_as_of)}`);
    if (Number.isInteger(company.filing_count)) bits.push(`공시 ${company.filing_count}건`);
    meta.textContent = bits.join(" · ");
    if (company.corpus_as_of) meta.title = company.corpus_as_of;
    if (bits.length) item.append(meta);
    corpusList.append(item);
  }
}

function renderPrompts(companies) {
  promptList.replaceChildren();
  const picks = exampleCompanies(companies);
  picks.forEach((company, index) => {
    const template = PROMPT_TEMPLATES[index];
    if (!template) return;
    const name = readableName(company.company_name);
    if (!name) return;
    addPrompt(template(name));
  });
}

function exampleCompanies(companies) {
  const named = companies.filter((company) => company.company_name);
  if (named.length <= 3) return named;
  const mid = Math.floor(named.length / 2);
  return [named[named.length - 1], named[mid], named[0]];
}

function readableName(value) {
  const text = String(value || "")
    .replace(/주식회사/g, "")
    .replace(/\(주\)/g, "")
    .replace(/㈜/g, "")
    .replace(/（주）/g, "")
    .trim();
  return text || String(value || "").trim();
}

function addPrompt(text) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = text;
  button.addEventListener("click", () => {
    if (busy) return;
    ask({ message: text, display: text });
  });
  promptList.append(button);
}

function ask({ message, display, clarification }) {
  if (busy) return;
  closeCoverage(false);
  closeHistory();
  ensureSession(display || message);
  emptyState.hidden = true;
  transcript.classList.add("has-thread");
  setBusy(true);
  appendUser(display || message);
  session.messages.push({
    role: "user",
    text: display || message,
    created_at: new Date().toISOString(),
  });
  persistSession();
  const assistant = appendAssistant();
  assistant.priorQuestion = clarification?.prior_message || message;
  activeAssistant = assistant;
  setActivity("질문을 보내고 있습니다");
  assistant.label = "질문을 보내고 있습니다";
  renderAssistant(assistant);
  const body = { message };
  if (clarification) body.clarification = clarification;
  streamAsk(body, assistant);
}

async function streamAsk(body, assistant) {
  abortController = new AbortController();
  let response;
  try {
    response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
      signal: abortController.signal,
    });
  } catch (error) {
    if (leaving || (error && error.name === "AbortError")) {
      setBusy(false);
      return;
    }
    fail(assistant, "서버에 연결하지 못했습니다. 잠시 후 다시 시도해 주세요.");
    return;
  }
  const contentType = response.headers.get("content-type") || "";
  if (!response.ok || !contentType.includes("text/event-stream")) {
    let message = "요청을 처리하지 못했습니다.";
    try {
      const payload = await response.json();
      if (payload && typeof payload.message === "string" && payload.message) {
        message = payload.message;
      }
    } catch {
      /* The body was not the public JSON error. */
    }
    fail(assistant, message);
    return;
  }
  try {
    await readSse(response, (eventName, data) => applyEvent(assistant, eventName, data));
  } catch (error) {
    if (leaving || (error && error.name === "AbortError")) {
      setBusy(false);
      return;
    }
    if (!assistant.closed) fail(assistant, "응답을 읽는 중 연결이 끊겼습니다.");
    else setBusy(false);
    return;
  }
  if (leaving) {
    setBusy(false);
    return;
  }
  if (!assistant.closed) {
    fail(assistant, "응답이 끝나기 전에 연결이 끊겼습니다.");
    return;
  }
  setBusy(false);
  setActivity("");
}

function applyEvent(assistant, eventName, data) {
  if (leaving) return;
  if (eventName === "agent.status" && data && ACTIVITY[data.status]) {
    setActivity(ACTIVITY[data.status]);
    if (assistant.kind === "pending") {
      assistant.label = ACTIVITY[data.status];
      renderAssistant(assistant);
    }
    return;
  }
  if (eventName === "answer.delta" && data && typeof data.text === "string") {
    appendDelta(assistant, data.text);
    renderAssistant(assistant);
    return;
  }
  if (eventName === "citation" && data && data.citation_id) {
    noteCitation(assistant, data);
    renderAssistant(assistant);
    return;
  }
  if (eventName === "run.completed" && data) {
    applyCompleted(assistant, data);
    renderAssistant(assistant);
    rememberAssistant(assistant);
    setActivity(assistant.label || "답변이 도착했습니다");
    return;
  }
  if (eventName === "error") {
    fail(assistant, (data && data.message) || "요청을 처리하지 못했습니다.", data && data.code);
  }
}

function fail(assistant, message, code) {
  if (assistant.closed || leaving) return;
  assistant.closed = true;
  assistant.kind = code === "request_timeout" ? "timeout" : "error";
  assistant.label = code === "request_timeout" ? "시간이 초과되었습니다" : "요청을 처리하지 못했습니다";
  assistant.retry = !code || RETRY_CODES.has(code);
  assistant.errorText = message;
  renderAssistant(assistant);
  rememberAssistant(assistant);
  setActivity(assistant.label);
  setBusy(false);
}

async function readSse(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    buffer = buffer.replace(/\r\n/g, "\n");
    let splitAt = buffer.indexOf("\n\n");
    while (splitAt !== -1) {
      const block = buffer.slice(0, splitAt);
      buffer = buffer.slice(splitAt + 2);
      const parsed = parseSseBlock(block);
      if (parsed) onEvent(parsed.event, parsed.data);
      splitAt = buffer.indexOf("\n\n");
    }
  }
}

function parseSseBlock(block) {
  let eventName = "message";
  const dataLines = [];
  for (const line of block.split("\n")) {
    if (!line || line.startsWith(":")) continue;
    if (line.startsWith("event:")) eventName = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  }
  if (!dataLines.length) return null;
  return { event: eventName, data: JSON.parse(dataLines.join("\n")) };
}

function appendUser(text) {
  const article = document.createElement("article");
  article.className = "turn";
  const kicker = document.createElement("p");
  kicker.className = "kicker";
  kicker.textContent = "질문";
  const heading = document.createElement("h2");
  heading.className = "ask-text";
  heading.textContent = text;
  article.append(kicker, heading);
  transcript.append(article);
  scrollTranscript(true);
  return article;
}

function appendAssistant() {
  const article = transcript.lastElementChild;
  const reply = document.createElement("div");
  reply.className = "reply pending";
  article.append(reply);
  const assistant = {
    node: reply,
    segments: [],
    citations: new Map(),
    kind: "pending",
    label: "질문을 보내고 있습니다",
    limitations: [],
    candidates: [],
    priorQuestion: "",
    errorText: "",
    retry: false,
    closed: false,
  };
  renderAssistant(assistant);
  scrollTranscript(true);
  return assistant;
}

function appendDelta(assistant, text) {
  const last = assistant.segments[assistant.segments.length - 1];
  if (last && last.type === "text") last.text += text;
  else assistant.segments.push({ type: "text", text });
}

function noteCitation(assistant, payload) {
  const id = String(payload.citation_id);
  const previous = assistant.citations.get(id) || {};
  assistant.citations.set(id, { ...previous, ...payload, citation_id: id });
  const already = assistant.segments.some((segment) => segment.type === "cite" && segment.id === id);
  const text = assistant.segments
    .filter((segment) => segment.type === "text")
    .map((segment) => segment.text)
    .join("");
  if (!already && !text.includes(`[${id}]`)) assistant.segments.push({ type: "cite", id });
}

function applyCompleted(assistant, data) {
  const answer = data.answer || {};
  const text = typeof answer.text === "string" ? answer.text : "";
  const bound = bindCitations(answer.citations || [], data.evidence || [], assistant.citations);
  assistant.citations = new Map(bound.map((item) => [String(item.citation_id), item]));
  assistant.closed = true;
  assistant.segments = text ? [{ type: "text", text }] : [];
  const shown = shownCitationIds(assistant);
  for (const id of assistant.citations.keys()) {
    if (!shown.has(id)) assistant.segments.push({ type: "cite", id });
  }
  const view = present(data);
  assistant.kind = view.kind;
  assistant.label = view.label;
  assistant.retry = Boolean(view.retry);
  assistant.limitations = scopeNotes(text, answer.limitations);
  assistant.candidates = data.clarification?.candidates || [];
  assistant.errorText = "";
}

function bindCitations(answerCitations, evidenceList, streamed) {
  const evidenceById = new Map(
    (Array.isArray(evidenceList) ? evidenceList : []).map((item) => [item.evidence_id, item]),
  );
  return (Array.isArray(answerCitations) ? answerCitations : []).map((item) => {
    const streamedItem = streamed.get(String(item.citation_id)) || {};
    const evidence = evidenceById.get(item.evidence_id) || {};
    return {
      ...streamedItem,
      ...evidence,
      citation_id: String(item.citation_id),
      evidence_id: item.evidence_id,
    };
  });
}

function shownCitationIds(assistant) {
  const shown = new Set();
  const text = assistant.segments
    .filter((segment) => segment.type === "text")
    .map((segment) => segment.text)
    .join("");
  for (const match of text.matchAll(/\[(\d+)\]/g)) {
    if (assistant.citations.has(match[1])) shown.add(match[1]);
  }
  for (const segment of assistant.segments) {
    if (segment.type === "cite") shown.add(segment.id);
  }
  return shown;
}

function present(payload) {
  const action = payload.action;
  const status = payload.status;
  const reason = payload.abstain_reason;
  const stop = payload.answer_stop;
  const cited = Array.isArray(payload.answer && payload.answer.citations)
    && payload.answer.citations.length > 0;
  if (action === "ask_clarification") {
    return { kind: "clarification", label: "확인이 필요합니다", retry: false };
  }
  if (
    status === "company_unsupported"
    || status === "unsupported_company"
    || reason === "unsupported_company"
  ) {
    return { kind: "unsupported", label: "지원 범위 밖", retry: false };
  }
  if (status === "retrieval_unavailable" || reason === "h10_unavailable") {
    return { kind: "unavailable", label: "검색을 마치지 못했습니다", retry: true };
  }
  if (stop === "generation_budget_fallback" || reason === "answer_budget_exhausted") {
    if (cited) return { kind: "partial", label: "일부만 확인했습니다", retry: true };
    return { kind: "limited", label: "답변을 마치지 못했습니다", retry: true };
  }
  if (status === "no_matching_filing" || reason === "no_matching_filing") {
    return { kind: "missing", label: "해당 조건의 공시를 찾지 못함", retry: false };
  }
  if (status === "corpus_not_fresh_enough" || reason === "corpus_not_fresh_enough") {
    return { kind: "limited", label: "데이터 기준 이후의 시점", retry: false };
  }
  if (status === "retrieval_no_result" || reason === "no_results") {
    return { kind: "empty", label: "해당하는 내용을 찾지 못함", retry: false };
  }
  if (reason === "insufficient_evidence" || status === "abstained") {
    return { kind: "abstained", label: "공시에서 확인하지 못함", retry: false };
  }
  if (action === "answer_directly") return { kind: "direct", label: "", retry: false };
  if (action === "search_disclosures" && status === "completed") {
    return { kind: "grounded", label: "공시 원문 기반", retry: false };
  }
  if (action === "abstain") return { kind: "abstained", label: "답변을 보류했습니다", retry: false };
  return { kind: "answer", label: "", retry: false };
}

function scopeNotes(answerText, limitations) {
  const answer = squash(answerText);
  const seen = new Set();
  const notes = [];
  const items = Array.isArray(limitations) ? limitations : [];
  for (const item of items) {
    const text = String(item || "").trim();
    const key = squash(text);
    if (!key || seen.has(key)) continue;
    seen.add(key);
    if (answer && (key === answer || answer.includes(key))) continue;
    notes.push(text);
  }
  return notes;
}

function squash(value) {
  return String(value || "").replace(/\s+/g, " ").trim();
}

function renderAssistant(assistant) {
  const node = assistant.node;
  node.className = `reply ${assistant.kind}`;
  node.replaceChildren();
  if (assistant.kind === "pending") {
    const status = document.createElement("p");
    status.className = "pending-status";
    status.setAttribute("role", "status");
    const dot = document.createElement("span");
    dot.className = "dot";
    dot.setAttribute("aria-hidden", "true");
    status.append(dot, document.createTextNode(assistant.label || "조사하고 있습니다"));
    node.append(status);
  } else {
    const who = document.createElement("p");
    who.className = "kicker";
    who.textContent = "답변";
    node.append(who);
    if (assistant.label) {
      const status = document.createElement("p");
      status.className = "reply-status";
      status.setAttribute("role", "status");
      status.textContent = assistant.label;
      node.append(status);
    }
  }
  const body = document.createElement("div");
  body.className = "message-body";
  for (const segment of assistant.segments) {
    if (segment.type === "text") appendLinkedText(body, segment.text, assistant.citations);
    else {
      const citation = assistant.citations.get(segment.id);
      if (citation) body.append(citationButton(citation));
    }
  }
  if (assistant.errorText) {
    const error = document.createElement("div");
    error.className = "error-line";
    error.setAttribute("role", "alert");
    error.textContent = assistant.errorText;
    body.append(error);
  }
  if (body.childNodes.length) node.append(body);
  if (assistant.kind === "grounded" || assistant.kind === "partial") {
    appendAnswerTables(node, assistant);
  }
  if (assistant.kind === "grounded" && assistant.citations.size) {
    const note = document.createElement("p");
    note.className = "evidence-note";
    const count = assistant.citations.size;
    note.textContent = `공시 원문 ${count}건 · 번호를 클릭하여 내용을 확인하세요.`;
    node.append(note);
  }
  if (assistant.limitations.length && INLINE_NOTES.has(assistant.kind)) {
    const notes = document.createElement("div");
    notes.className = "limitations";
    for (const note of assistant.limitations) {
      const line = document.createElement("p");
      line.textContent = note;
      notes.append(line);
    }
    node.append(notes);
  } else if (assistant.limitations.length) {
    const details = document.createElement("details");
    details.className = "scope";
    const summary = document.createElement("summary");
    summary.textContent = "답변 범위";
    details.append(summary);
    for (const note of assistant.limitations) {
      const line = document.createElement("div");
      line.textContent = note;
      details.append(line);
    }
    node.append(details);
  }
  if (assistant.retry && assistant.closed && assistant.priorQuestion) {
    const retry = document.createElement("button");
    retry.type = "button";
    retry.className = "retry";
    retry.textContent = "다시 질문하기";
    retry.addEventListener("click", () => {
      if (busy || !assistant.priorQuestion) return;
      ask({ message: assistant.priorQuestion, display: assistant.priorQuestion });
    });
    node.append(retry);
  }
  if (assistant.candidates.length) {
    const choices = document.createElement("div");
    choices.className = "candidates";
    for (const candidate of assistant.candidates) {
      if (!candidate || !candidate.company_name) continue;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "candidate";
      button.textContent = candidate.stock_code
        ? `${candidate.company_name} (${candidate.stock_code})`
        : candidate.company_name;
      button.disabled = busy;
      button.addEventListener("click", () => {
        ask({
          message: candidate.company_name,
          display: candidate.company_name,
          clarification: {
            prior_message: assistant.priorQuestion,
            selected_company_name: candidate.company_name,
          },
        });
      });
      choices.append(button);
    }
    node.append(choices);
  }
  scrollTranscript(false);
}

function appendLinkedText(container, text, citations) {
  const pattern = /\[(\d+)\]/g;
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    container.append(document.createTextNode(text.slice(last, match.index)));
    const citation = citations.get(match[1]);
    if (citation) container.append(citationButton(citation));
    else container.append(document.createTextNode(match[0]));
    last = match.index + match[0].length;
  }
  container.append(document.createTextNode(text.slice(last)));
}

function appendAnswerTables(node, assistant) {
  for (const citation of assistant.citations.values()) {
    const blocks = Array.isArray(citation.blocks) ? citation.blocks : [];
    const tables = blocks.filter(
      (block) => block && block.type === "table" && Array.isArray(block.cells) && block.cells.length,
    );
    if (!tables.length) continue;
    const wrap = document.createElement("div");
    wrap.className = "answer-tables";
    const caption = document.createElement("p");
    caption.className = "answer-table-caption";
    caption.textContent = `원문 ${citation.citation_id}의 표`;
    wrap.append(caption);
    for (const block of tables) {
      const table = renderTable(block);
      if (table) wrap.append(table);
    }
    if (wrap.querySelector("table")) node.append(wrap);
  }
}

function citationButton(citation) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "cite";
  button.textContent = `[${citation.citation_id}]`;
  button.setAttribute("aria-label", `원문 ${citation.citation_id} 보기`);
  button.dataset.citationId = String(citation.citation_id);
  button.dataset.evidenceId = String(citation.evidence_id || "");
  button.addEventListener("click", () => openDrawer(citation, button));
  return button;
}

function openDrawer(citation, trigger) {
  closeCoverage(false);
  closeHistory();
  drawerTrigger = trigger || null;
  drawerKicker.textContent = `원문 ${citation.citation_id}`;
  drawerTitle.textContent = shown(citation.company_name) || "공시 원문";
  const documentLine = filingLine(citation);
  drawerDocument.textContent = documentLine;
  drawerDocument.hidden = !documentLine;
  renderSource(citation);
  drawerMeta.replaceChildren();
  addFact(drawerMeta, "공시일", formatReceiptDate(citation.receipt_date));
  addFact(drawerMeta, "보고기간", citation.filing_period);
  const section = Array.isArray(citation.section_path)
    ? citation.section_path.filter(Boolean).join(" · ")
    : citation.section_path;
  addFact(drawerMeta, "섹션", typeof section === "string" ? squash(section) : section);
  addFact(drawerMeta, "접수번호", citation.receipt_no);
  drawerMeta.hidden = drawerMeta.childElementCount === 0;
  document.querySelector("#drawer-facts-label").hidden = drawerMeta.hidden;

  drawerTechnical.replaceChildren();
  addFact(drawerTechnical, "ID", citation.evidence_id, { evidenceId: true });
  const lines = locatorLines(citation.source_locator);
  if (!lines.length && Array.isArray(citation.element_ids)) {
    const fallback = citation.element_ids.map(String).filter((item) => item && !isPrivatePath(item));
    if (fallback.length) lines.push(fallback.join(", "));
  }
  if (lines.length) addFact(drawerTechnical, "원문 위치", lines.join("\n"));
  drawerMore.hidden = drawerTechnical.childElementCount === 0;
  drawerMore.open = false;
  scrim.hidden = false;
  drawer.hidden = false;
  drawer.scrollTop = 0;
  document.body.dataset.drawer = "open";
  document.querySelector("#drawer-close").focus();
}

function renderSource(citation) {
  drawerText.replaceChildren();
  const blocks = Array.isArray(citation.blocks) ? citation.blocks : [];
  const tableExpected = locatorHasTable(citation);
  const wrote = renderBlocks(drawerText, blocks, tableExpected);
  if (!wrote) {
    const raw = typeof citation.text === "string" && citation.text ? citation.text : citation.excerpt;
    const body = evidenceProse(raw);
    if (body) drawerText.append(proseNode(body));
  }
  drawerText.hidden = drawerText.childElementCount === 0;
  document.querySelector("#drawer-source-label").hidden = drawerText.hidden;
}

function locatorHasTable(citation) {
  return Array.isArray(citation.source_locator)
    && citation.source_locator.some((span) => span && span.kind === "table");
}

function renderBlocks(container, blocks, tableExpected) {
  const tables = blocks.filter(
    (block) => block && block.type === "table" && Array.isArray(block.cells) && block.cells.length,
  );
  if (tableExpected && !tables.length) return false;
  if (!blocks.length) return false;
  let wrote = false;
  for (const block of blocks) {
    if (!block || typeof block !== "object") continue;
    if (block.type === "table") {
      const table = renderTable(block);
      if (table) {
        container.append(table);
        wrote = true;
      }
    } else if (block.type === "text" && typeof block.text === "string" && block.text.trim()) {
      container.append(proseNode(block.text.trim()));
      wrote = true;
    }
  }
  return wrote;
}

function renderTable(block) {
  const cells = (Array.isArray(block.cells) ? block.cells : []).filter(validCell);
  if (!cells.length) return null;
  const rowCount = Number.isInteger(block.row_count) && block.row_count > 0
    ? block.row_count
    : Math.max(...cells.map((cell) => cell.start_row + Math.max(cell.rowspan || 1, 1)));
  const rows = [];
  for (let index = 0; index < rowCount; index += 1) rows.push(document.createElement("tr"));
  const headerRows = new Set();
  for (let index = 0; index < rowCount; index += 1) {
    const starters = cells.filter((cell) => cell.start_row === index);
    if (starters.length && starters.every((cell) => cell.header)) headerRows.add(index);
  }
  for (const cell of cells) {
    if (cell.start_row < 0 || cell.start_row >= rows.length) continue;
    const node = document.createElement(cell.header ? "th" : "td");
    node.textContent = String(cell.text || "");
    if (cell.header && (cell.rowspan || 1) === 1) node.scope = "col";
    if ((cell.rowspan || 1) > 1) node.rowSpan = cell.rowspan;
    if ((cell.colspan || 1) > 1) node.colSpan = cell.colspan;
    rows[cell.start_row].append(node);
  }
  const table = document.createElement("table");
  table.className = "source-table";
  const thead = document.createElement("thead");
  const tbody = document.createElement("tbody");
  let split = 0;
  while (split < rows.length && headerRows.has(split)) split += 1;
  rows.forEach((row, index) => {
    if (index < split) thead.append(row);
    else tbody.append(row);
  });
  if (thead.childElementCount) table.append(thead);
  if (tbody.childElementCount) table.append(tbody);
  const scroller = document.createElement("div");
  scroller.className = "table-scroll";
  scroller.append(table);
  return scroller;
}

function validCell(cell) {
  return Boolean(cell)
    && typeof cell === "object"
    && Number.isInteger(cell.start_row)
    && Number.isInteger(cell.start_col);
}

function proseNode(text) {
  const paragraph = document.createElement("p");
  paragraph.className = "source-prose";
  paragraph.textContent = text;
  return paragraph;
}

function evidenceProse(value) {
  if (typeof value !== "string" || isPrivatePath(value)) return "";
  const stripped = value.replace(/^\[section\][^\n]*\n+/, "").trim();
  return stripped || value.trim();
}

function filingLine(citation) {
  const report = shown(citation.report_name) || shown(citation.document_name);
  const period = shown(citation.filing_period);
  if (report && period && !report.includes(period)) return `${report} (${period})`;
  return report || period;
}

function addFact(list, label, value, options = {}) {
  const text = shown(value);
  if (!text) return;
  const term = document.createElement("dt");
  term.textContent = label;
  const detail = document.createElement("dd");
  detail.textContent = text;
  if (options.evidenceId) detail.dataset.evidenceId = text;
  if (label === "ID") detail.dataset.evidenceId = text;
  list.append(term, detail);
}

function locatorLines(locator) {
  if (!Array.isArray(locator)) return [];
  const lines = [];
  for (const span of locator) {
    if (!span || typeof span !== "object") continue;
    const bits = [];
    if (span.element_id && !isPrivatePath(span.element_id)) bits.push(String(span.element_id));
    if (span.kind && !isPrivatePath(span.kind)) bits.push(String(span.kind));
    const nested = span.element_source_locator;
    if (nested && typeof nested === "object") {
      if (nested.member_name && !isPrivatePath(nested.member_name)) bits.push(String(nested.member_name));
      if (Array.isArray(nested.path)) {
        const tags = nested.path.map((step) => step && step.tag).filter(Boolean);
        if (tags.length) bits.push(tags.join(" / "));
      }
    }
    if (bits.length) lines.push(bits.join(" · "));
  }
  return lines;
}

function closeDrawer() {
  if (drawer.hidden) return;
  drawer.hidden = true;
  scrim.hidden = true;
  delete document.body.dataset.drawer;
  if (drawerTrigger) drawerTrigger.focus();
  drawerTrigger = null;
}

function openCoverage(trigger) {
  if (!drawer.hidden) closeDrawer();
  closeHistory();
  coverageTrigger = trigger || coverageToggle;
  coveragePanel.hidden = false;
  coverageToggle.setAttribute("aria-expanded", "true");
  coverageFilter.focus();
}

function closeCoverage(restoreFocus = true) {
  if (coveragePanel.hidden) return;
  coveragePanel.hidden = true;
  coverageToggle.setAttribute("aria-expanded", "false");
  if (restoreFocus && coverageTrigger) coverageTrigger.focus();
  coverageTrigger = null;
}

function startNewConversation() {
  leaveCurrent();
  session = null;
  activeAssistant = null;
  clearThread();
  closeDrawer();
  closeCoverage(false);
  closeHistory();
  renderHistory();
  questionInput.focus();
}

function leaveCurrent() {
  leaving = true;
  if (activeAssistant && !activeAssistant.closed) {
    activeAssistant.closed = true;
    const text = assistantText(activeAssistant);
    if (session && text) {
      if (activeAssistant.kind === "pending") {
        activeAssistant.kind = "answer";
        activeAssistant.label = "";
      }
      rememberAssistant(activeAssistant);
    }
  }
  if (abortController) abortController.abort();
  abortController = null;
  activeAssistant = null;
  leaving = false;
  setBusy(false);
  setActivity("");
}

function clearThread() {
  for (const turn of transcript.querySelectorAll(".turn")) turn.remove();
  emptyState.hidden = false;
  transcript.classList.remove("has-thread");
}

function ensureSession(question) {
  if (session) return session;
  const now = new Date().toISOString();
  session = {
    id: crypto.randomUUID(),
    title: store ? store.conversationTitle(question) : String(question || "").slice(0, 42),
    created_at: now,
    updated_at: now,
    messages: [],
  };
  return session;
}

function assistantText(assistant) {
  return assistant.segments
    .filter((segment) => segment.type === "text")
    .map((segment) => segment.text)
    .join("");
}

function rememberAssistant(assistant) {
  if (!session || !assistant) return;
  session.messages.push({
    role: "assistant",
    text: assistantText(assistant),
    created_at: new Date().toISOString(),
    kind: assistant.kind,
    label: assistant.label,
    limitations: assistant.limitations,
    errorText: assistant.errorText,
    priorQuestion: assistant.priorQuestion,
    retry: Boolean(assistant.retry),
    candidates: assistant.candidates,
    citations: [...assistant.citations.values()],
  });
  persistSession();
}

function persistSession() {
  if (!session || !store) return;
  session.updated_at = new Date().toISOString();
  store.createConversationStore().save(session).then(() => renderHistory()).catch(() => {
    showHistoryFailure("이 브라우저에 대화를 저장하지 못했습니다.");
  });
}

async function renderHistory() {
  if (!store) {
    showHistoryFailure("이 브라우저에서는 대화를 저장할 수 없습니다.");
    return;
  }
  let records = [];
  try {
    records = await store.createConversationStore().list();
  } catch {
    showHistoryFailure("대화를 불러오지 못했습니다.");
    return;
  }
  historyList.replaceChildren();
  if (!records.length) {
    const empty = document.createElement("p");
    empty.className = "history-empty";
    empty.textContent = "아직 대화가 없습니다.";
    historyList.append(empty);
    return;
  }
  for (const group of groupConversations(records)) {
    const section = document.createElement("section");
    const heading = document.createElement("h2");
    heading.className = "history-day";
    heading.textContent = group.label;
    const list = document.createElement("ul");
    for (const record of group.items) {
      const item = document.createElement("li");
      const open = document.createElement("button");
      open.type = "button";
      open.className = "history-item";
      open.textContent = record.title || "새 대화";
      if (session && session.id === record.id) open.setAttribute("aria-current", "true");
      open.addEventListener("click", () => openStored(record.id));
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "history-delete";
      remove.textContent = "삭제";
      remove.setAttribute("aria-label", `${record.title || "대화"} 삭제`);
      remove.addEventListener("click", (event) => {
        event.stopPropagation();
        deleteConversation(record.id);
      });
      item.append(open, remove);
      list.append(item);
    }
    section.append(heading, list);
    historyList.append(section);
  }
}

function groupConversations(records) {
  const groups = [];
  const byLabel = new Map();
  for (const record of records) {
    const label = dayLabel(record.updated_at || record.created_at);
    let group = byLabel.get(label);
    if (!group) {
      group = { label, items: [] };
      byLabel.set(label, group);
      groups.push(group);
    }
    group.items.push(record);
  }
  return groups;
}

function dayLabel(iso) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "이전";
  const start = (value) => new Date(value.getFullYear(), value.getMonth(), value.getDate()).getTime();
  const diff = Math.round((start(new Date()) - start(date)) / 86400000);
  if (diff === 0) return "오늘";
  if (diff === 1) return "어제";
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${date.getFullYear()}.${month}.${day}`;
}

async function openStored(id) {
  if (!store) return;
  if (session && session.id === id && transcript.querySelector(".turn")) {
    closeHistory();
    return;
  }
  let record;
  try {
    record = await store.createConversationStore().get(id);
  } catch {
    showHistoryFailure("대화를 불러오지 못했습니다.");
    return;
  }
  if (!record) {
    renderHistory();
    return;
  }
  leaveCurrent();
  session = record;
  clearThread();
  closeDrawer();
  closeCoverage(false);
  closeHistory();
  emptyState.hidden = true;
  transcript.classList.add("has-thread");
  for (const message of record.messages || []) renderStoredMessage(message);
  renderHistory();
  questionInput.focus();
}

function renderStoredMessage(message) {
  if (!message || message.role === "user") {
    appendUser(message ? message.text : "");
    return;
  }
  const article = transcript.lastElementChild;
  if (!article) return;
  const reply = document.createElement("div");
  article.append(reply);
  const citations = new Map(
    (message.citations || []).map((item) => [String(item.citation_id), item]),
  );
  const assistant = {
    node: reply,
    segments: message.text ? [{ type: "text", text: message.text }] : [],
    citations,
    kind: message.kind || "answer",
    label: message.label || "",
    limitations: message.limitations || [],
    candidates: message.candidates || [],
    priorQuestion: message.priorQuestion || "",
    errorText: message.errorText || "",
    retry: Boolean(message.retry),
    closed: true,
  };
  const shown = shownCitationIds(assistant);
  for (const id of citations.keys()) {
    if (!shown.has(id)) assistant.segments.push({ type: "cite", id });
  }
  renderAssistant(assistant);
}

async function deleteConversation(id) {
  if (!store) return;
  try {
    await store.createConversationStore().delete(id);
  } catch {
    showHistoryFailure("대화를 삭제하지 못했습니다.");
    return;
  }
  if (session && session.id === id) {
    session = null;
    activeAssistant = null;
    clearThread();
    closeDrawer();
  }
  renderHistory();
}

function showHistoryFailure(message) {
  historyList.replaceChildren();
  const note = document.createElement("p");
  note.className = "history-empty";
  note.textContent = message;
  historyList.append(note);
}

function openHistory() {
  if (!drawer.hidden) closeDrawer();
  closeCoverage(false);
  document.body.dataset.history = "open";
  historyOpen.setAttribute("aria-expanded", "true");
  historyDismiss.hidden = false;
}

function closeHistory() {
  if (document.body.dataset.history !== "open") return;
  delete document.body.dataset.history;
  historyOpen.setAttribute("aria-expanded", "false");
  historyDismiss.hidden = true;
}

function formatAsOf(iso) {
  const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}:\d{2})/.exec(String(iso));
  if (!match) return String(iso);
  return `${match[1]}.${match[2]}.${match[3]} ${match[4]}`;
}

function formatReceiptDate(value) {
  const text = String(value || "");
  const match = /^(\d{4})(\d{2})(\d{2})$/.exec(text);
  if (!match) return text;
  return `${match[1]}.${match[2]}.${match[3]}`;
}

function shown(value) {
  if (value === undefined || value === null) return "";
  const text = String(value).trim();
  if (!text || isPrivatePath(text)) return "";
  return text;
}

function isPrivatePath(value) {
  return /[A-Za-z]:\\|\\\\|data[\\/](?:product|parsed|raw|cache)/.test(String(value));
}

function setActivity(text) {
  activity.textContent = text || "";
}

function setBusy(value) {
  busy = value;
  form.setAttribute("aria-busy", value ? "true" : "false");
  updateSendState();
  for (const button of promptList.querySelectorAll("button")) button.disabled = value;
  for (const button of transcript.querySelectorAll(".candidate, .retry")) button.disabled = value;
}

function updateSendState() {
  const empty = !questionInput.value.trim();
  sendButton.disabled = busy || empty;
  sendButton.textContent = busy ? "조사 중" : "질문하기";
}

function resizeQuestion() {
  questionInput.style.height = "auto";
  const next = Math.min(Math.max(questionInput.scrollHeight, 52), 176);
  questionInput.style.height = `${next}px`;
}

function scrollTranscript(force) {
  const distance = transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight;
  if (force || distance < 140) transcript.scrollTop = transcript.scrollHeight;
}

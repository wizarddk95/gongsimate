const ACTIVITY = {
  resolving_company: "기업을 확인하는 중",
  searching_disclosures: "공시를 검색하는 중",
  generating_answer: "답변을 작성하는 중",
};

const transcript = document.querySelector("#transcript");
const form = document.querySelector("#ask-form");
const questionInput = document.querySelector("#question");
const sendButton = document.querySelector("#send");
const activity = document.querySelector("#activity");
const corpusDescription = document.querySelector("#corpus-description");
const corpusList = document.querySelector("#corpus-list");
const promptList = document.querySelector("#prompt-list");
const drawer = document.querySelector("#drawer");
const scrim = document.querySelector("#scrim");
const drawerMeta = document.querySelector("#drawer-meta");
const drawerText = document.querySelector("#drawer-text");
const drawerLocator = document.querySelector("#drawer-locator");
const drawerTitle = document.querySelector("#drawer-title");

let busy = false;
let drawerTrigger = null;

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = questionInput.value.trim();
  if (!message || busy) return;
  questionInput.value = "";
  ask({ message, display: message });
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelector("#drawer-close").addEventListener("click", closeDrawer);
scrim.addEventListener("click", closeDrawer);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeDrawer();
});

loadCorpus();

async function loadCorpus() {
  try {
    const response = await fetch("/api/corpus");
    if (!response.ok) throw new Error("corpus");
    renderCorpus(await response.json());
  } catch {
    corpusDescription.textContent = "지원 기업 목록을 불러오지 못했습니다.";
    corpusList.replaceChildren();
    promptList.replaceChildren();
  }
}

function renderCorpus(payload) {
  corpusDescription.textContent = payload.description || "색인된 공시 원문만 검색합니다.";
  corpusList.replaceChildren();
  const companies = Array.isArray(payload.companies) ? payload.companies : [];
  if (!companies.length) {
    const item = document.createElement("li");
    item.textContent = "현재 색인된 기업이 없습니다.";
    corpusList.append(item);
  }
  for (const company of companies) {
    const item = document.createElement("li");
    const name = document.createElement("strong");
    name.textContent = company.company_name || "이름 없는 기업";
    item.append(name);
    if (company.corpus_as_of) {
      const asOf = document.createElement("span");
      asOf.textContent = `데이터 기준 ${formatAsOf(company.corpus_as_of)}`;
      asOf.title = company.corpus_as_of;
      item.append(asOf);
    }
    if (Number.isInteger(company.filing_count)) {
      const count = document.createElement("span");
      count.textContent = `색인 공시 ${company.filing_count}건`;
      item.append(count);
    }
    corpusList.append(item);
  }
  renderPrompts(companies);
}

function renderPrompts(companies) {
  promptList.replaceChildren();
  for (const company of companies) {
    if (!company.company_name) continue;
    addPrompt(`${company.company_name}의 주요 사업은?`);
  }
  addPrompt("안녕");
  addPrompt("RAG가 뭐야?");
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
  setBusy(true);
  appendUser(display || message);
  const assistant = appendAssistant();
  assistant.priorQuestion = clarification?.prior_message || message;
  activity.textContent = "요청을 보내는 중";
  const body = { message };
  if (clarification) body.clarification = clarification;
  streamAsk(body, assistant);
}

async function streamAsk(body, assistant) {
  let response;
  try {
    response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
    });
  } catch {
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
  } catch {
    if (!assistant.closed) fail(assistant, "응답을 읽는 중 연결이 끊겼습니다.");
    else setBusy(false);
    return;
  }
  if (!assistant.closed) {
    fail(assistant, "응답이 끝나기 전에 연결이 끊겼습니다.");
    return;
  }
  setBusy(false);
  activity.textContent = "";
}

function applyEvent(assistant, eventName, data) {
  if (eventName === "agent.status" && data && ACTIVITY[data.status]) {
    activity.textContent = ACTIVITY[data.status];
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
    activity.textContent = "";
    return;
  }
  if (eventName === "error") {
    fail(assistant, (data && data.message) || "요청을 처리하지 못했습니다.");
  }
}

function fail(assistant, message) {
  if (assistant.closed) return;
  assistant.closed = true;
  assistant.kind = "error";
  assistant.label = "오류";
  assistant.errorText = message;
  renderAssistant(assistant);
  activity.textContent = "";
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
  article.className = "message user";
  article.textContent = text;
  transcript.append(article);
  scrollTranscript();
}

function appendAssistant() {
  const article = document.createElement("article");
  article.className = "message assistant pending";
  transcript.append(article);
  const assistant = {
    node: article,
    segments: [],
    citations: new Map(),
    kind: "pending",
    label: "실행 중",
    limitations: [],
    candidates: [],
    priorQuestion: "",
    errorText: "",
    closed: false,
  };
  renderAssistant(assistant);
  scrollTranscript();
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
  const streamed = assistant.segments
    .filter((segment) => segment.type === "text")
    .map((segment) => segment.text)
    .join("");
  if (streamed !== text) assistant.segments = text ? [{ type: "text", text }] : [];
  const shown = shownCitationIds(assistant);
  for (const id of assistant.citations.keys()) {
    if (!shown.has(id)) assistant.segments.push({ type: "cite", id });
  }
  assistant.closed = true;
  const view = present(data);
  assistant.kind = view.kind;
  assistant.label = view.label;
  assistant.limitations = Array.isArray(answer.limitations) ? answer.limitations.filter(Boolean) : [];
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
    return { ...streamedItem, ...evidence, citation_id: String(item.citation_id), evidence_id: item.evidence_id };
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
  if (action === "ask_clarification") {
    if (status === "company_ambiguous") return { kind: "clarification", label: "어느 기업인지 확인이 필요합니다" };
    if (status === "company_not_found") return { kind: "clarification", label: "기업을 특정하지 못했습니다" };
    return { kind: "clarification", label: "확인이 필요합니다" };
  }
  if (status === "company_unsupported" || status === "unsupported_company" || reason === "unsupported_company") {
    return { kind: "unsupported", label: "현재 색인에 없는 기업" };
  }
  if (status === "no_matching_filing" || reason === "no_matching_filing") {
    return { kind: "abstain", label: "조건에 맞는 공시 없음" };
  }
  if (reason === "insufficient_evidence" || status === "abstained") {
    return { kind: "abstain", label: "근거 부족" };
  }
  if (status === "corpus_not_fresh_enough" || reason === "corpus_not_fresh_enough") {
    return { kind: "abstain", label: "색인 기준 시각 이후" };
  }
  if (action === "answer_directly") return { kind: "direct", label: "바로 답변" };
  if (action === "search_disclosures" && status === "completed") return { kind: "grounded", label: "공시 근거" };
  if (action === "abstain") return { kind: "abstain", label: "답변을 보류했습니다" };
  return { kind: "answer", label: "답변" };
}

function renderAssistant(assistant) {
  const node = assistant.node;
  node.className = `message assistant ${assistant.kind}`;
  node.replaceChildren();
  if (assistant.label) {
    const badge = document.createElement("div");
    badge.className = "badge";
    badge.textContent = assistant.label;
    node.append(badge);
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
    error.textContent = assistant.errorText;
    body.append(error);
  }
  node.append(body);
  const answerText = assistant.segments
    .filter((segment) => segment.type === "text")
    .map((segment) => segment.text)
    .join("");
  const notes = assistant.limitations.filter((item) => item !== answerText);
  if (notes.length) {
    const list = document.createElement("div");
    list.className = "notes";
    for (const note of notes) {
      const line = document.createElement("div");
      line.textContent = note;
      list.append(line);
    }
    node.append(list);
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
  scrollTranscript();
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

function citationButton(citation) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "cite";
  button.textContent = `[${citation.citation_id}]`;
  button.dataset.citationId = String(citation.citation_id);
  button.dataset.evidenceId = String(citation.evidence_id || "");
  button.addEventListener("click", () => openDrawer(citation, button));
  return button;
}

function openDrawer(citation, trigger) {
  drawerTrigger = trigger || null;
  drawerTitle.textContent = `근거 [${citation.citation_id}]`;
  drawerMeta.replaceChildren();
  addMeta("인용 번호", citation.citation_id);
  addMeta("근거 ID", citation.evidence_id);
  addMeta("회사", citation.company_name);
  addMeta("문서", citation.document_name);
  addMeta("보고서", citation.report_name);
  addMeta("접수번호", citation.receipt_no);
  addMeta("접수일", formatReceiptDate(citation.receipt_date));
  addMeta("보고기간", citation.filing_period);
  const section = Array.isArray(citation.section_path)
    ? citation.section_path.filter(Boolean).join(" > ")
    : citation.section_path;
  addMeta("섹션", section);
  drawerText.replaceChildren();
  const evidenceText = typeof citation.text === "string" && citation.text
    ? citation.text
    : citation.excerpt;
  if (typeof evidenceText === "string" && evidenceText) {
    drawerText.textContent = evidenceText;
  }
  drawerLocator.replaceChildren();
  const lines = locatorLines(citation.source_locator);
  if (!lines.length && Array.isArray(citation.element_ids) && citation.element_ids.length) {
    lines.push(citation.element_ids.map(String).filter((item) => !isPrivatePath(item)).join(", "));
  }
  if (lines.length) {
    const heading = document.createElement("div");
    heading.textContent = "위치";
    drawerLocator.append(heading);
    for (const line of lines) {
      const row = document.createElement("div");
      row.textContent = line;
      drawerLocator.append(row);
    }
  }
  scrim.hidden = false;
  drawer.hidden = false;
  document.querySelector("#drawer-close").focus();
}

function addMeta(label, value) {
  if (value === undefined || value === null || value === "") return;
  const text = String(value);
  if (isPrivatePath(text)) return;
  const term = document.createElement("dt");
  term.textContent = label;
  const detail = document.createElement("dd");
  detail.textContent = text;
  if (label === "근거 ID") detail.dataset.evidenceId = text;
  if (label === "인용 번호") detail.dataset.citationId = text;
  drawerMeta.append(term, detail);
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
  drawer.hidden = true;
  scrim.hidden = true;
  if (drawerTrigger) drawerTrigger.focus();
  drawerTrigger = null;
}

function formatAsOf(iso) {
  const match = /^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})/.exec(String(iso));
  if (!match) return String(iso);
  return `${match[1]} ${match[2]} KST`;
}

function formatReceiptDate(value) {
  const text = String(value || "");
  const match = /^(\d{4})(\d{2})(\d{2})$/.exec(text);
  if (!match) return text;
  return `${match[1]}-${match[2]}-${match[3]}`;
}

function isPrivatePath(value) {
  return /[A-Za-z]:\\|\\\\|data[\\/](?:product|parsed|raw|cache)/.test(String(value));
}

function setBusy(value) {
  busy = value;
  sendButton.disabled = value;
  questionInput.disabled = value;
  for (const button of promptList.querySelectorAll("button")) button.disabled = value;
  for (const button of transcript.querySelectorAll(".candidate")) button.disabled = value;
}

function scrollTranscript() {
  transcript.scrollTop = transcript.scrollHeight;
}

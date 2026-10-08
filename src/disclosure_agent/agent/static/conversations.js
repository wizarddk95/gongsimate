(function (root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (root) root.GongsiMateStore = api;
})(typeof window !== "undefined" ? window : globalThis, function () {
  const DB_NAME = "gongsimate";
  const STORE = "conversations";
  const TITLE_LIMIT = 42;

  function conversationTitle(question) {
    let text = String(question || "").replace(/\s+/g, " ").trim();
    text = text.replace(/[?？!！.。]+$/u, "").trim();
    const tails = [
      /\s*(?:은|는|이|가)?\s*무엇(?:인가|인지|이에요|예요)?$/,
      /\s*(?:은|는|이|가)?\s*뭐야$/,
      /\s*(?:을|를)?\s*설명해\s*주세요$/,
      /\s*(?:을|를)?\s*설명해주세요$/,
      /\s*(?:을|를)?\s*알려\s*주세요$/,
      /\s*(?:을|를)?\s*알려주세요$/,
    ];
    let changed = true;
    while (changed) {
      changed = false;
      for (const pattern of tails) {
        const next = text.replace(pattern, "").trim();
        if (next && next !== text) {
          text = next;
          changed = true;
        }
      }
    }
    if (text.length > TITLE_LIMIT) text = `${text.slice(0, TITLE_LIMIT).trimEnd()}…`;
    return text || "새 대화";
  }

  function sanitizeCitation(citation) {
    const source = citation && typeof citation === "object" ? citation : {};
    const row = {};
    copyString(row, source, "citation_id");
    copyString(row, source, "evidence_id");
    copyString(row, source, "company_name");
    copyString(row, source, "report_name");
    copyString(row, source, "document_name");
    copyString(row, source, "filing_period");
    copyString(row, source, "receipt_date");
    copyString(row, source, "receipt_no");
    copyString(row, source, "text");
    if (Array.isArray(source.section_path)) {
      row.section_path = source.section_path.map((item) => String(item));
    }
    if (Array.isArray(source.element_ids)) {
      row.element_ids = source.element_ids.map((item) => String(item));
    }
    row.blocks = sanitizeBlocks(source.blocks);
    if (Array.isArray(source.source_locator)) {
      row.source_locator = source.source_locator.map(sanitizeLocator);
    }
    return row;
  }

  function sanitizeConversation(conversation) {
    const source = conversation && typeof conversation === "object" ? conversation : {};
    const messages = Array.isArray(source.messages) ? source.messages.map(sanitizeMessage) : [];
    const firstQuestion = messages.find((item) => item.role === "user");
    return {
      id: String(source.id || ""),
      title: String(source.title || conversationTitle(firstQuestion ? firstQuestion.text : "")),
      created_at: String(source.created_at || ""),
      updated_at: String(source.updated_at || ""),
      messages,
    };
  }

  function sanitizeMessage(message) {
    const source = message && typeof message === "object" ? message : {};
    const row = {
      role: source.role === "assistant" ? "assistant" : "user",
      text: String(source.text || ""),
      created_at: String(source.created_at || ""),
    };
    if (row.role !== "assistant") return row;
    row.kind = String(source.kind || "");
    row.label = String(source.label || "");
    row.limitations = Array.isArray(source.limitations)
      ? source.limitations.map((item) => String(item))
      : [];
    row.errorText = String(source.errorText || "");
    row.priorQuestion = String(source.priorQuestion || "");
    row.retry = source.retry === true;
    row.candidates = Array.isArray(source.candidates)
      ? source.candidates.filter((item) => item && item.company_name).map((item) => ({
        company_name: String(item.company_name),
        stock_code: String(item.stock_code || ""),
      }))
      : [];
    row.citations = Array.isArray(source.citations) ? source.citations.map(sanitizeCitation) : [];
    return row;
  }

  function sanitizeBlocks(blocks) {
    if (!Array.isArray(blocks)) return [];
    const kept = [];
    for (const block of blocks) {
      if (!block || typeof block !== "object") continue;
      if (block.type === "text") {
        kept.push({ type: "text", text: String(block.text || "") });
        continue;
      }
      if (block.type !== "table" || !Array.isArray(block.cells)) continue;
      kept.push({
        type: "table",
        row_count: integer(block.row_count),
        col_count: integer(block.col_count),
        cells: block.cells.filter((cell) => cell && typeof cell === "object").map((cell) => ({
          start_row: integer(cell.start_row),
          start_col: integer(cell.start_col),
          rowspan: integer(cell.rowspan) || 1,
          colspan: integer(cell.colspan) || 1,
          text: String(cell.text || ""),
          header: Boolean(cell.header),
        })),
      });
    }
    return kept;
  }

  function sanitizeLocator(span) {
    if (!span || typeof span !== "object") return {};
    const row = {};
    for (const key of ["element_id", "kind", "role", "document_id"]) {
      if (span[key] !== undefined && span[key] !== null) row[key] = String(span[key]);
    }
    if (Number.isInteger(span.char_start)) row.char_start = span.char_start;
    if (Number.isInteger(span.char_end)) row.char_end = span.char_end;
    if (Number.isInteger(span.order_index)) row.order_index = span.order_index;
    const nested = span.element_source_locator;
    if (nested && typeof nested === "object") {
      const clean = {};
      if (nested.member_name) clean.member_name = String(nested.member_name);
      if (Array.isArray(nested.path)) {
        clean.path = nested.path.map((step) => ({
          tag: step && step.tag ? String(step.tag) : "",
          index: step && Number.isInteger(step.index) ? step.index : null,
        }));
      }
      row.element_source_locator = clean;
    }
    return row;
  }

  function copyString(target, source, key) {
    if (source[key] !== undefined && source[key] !== null) target[key] = String(source[key]);
  }

  function integer(value) {
    return Number.isInteger(value) ? value : 0;
  }

  function createConversationStore() {
    function open() {
      return new Promise((resolve, reject) => {
        if (typeof indexedDB === "undefined") {
          reject(new Error("indexedDB unavailable"));
          return;
        }
        const request = indexedDB.open(DB_NAME, 1);
        request.onupgradeneeded = () => {
          const db = request.result;
          if (!db.objectStoreNames.contains(STORE)) {
            db.createObjectStore(STORE, { keyPath: "id" });
          }
        };
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error || new Error("indexedDB open failed"));
      });
    }

    function run(mode, action) {
      return open().then((db) => new Promise((resolve, reject) => {
        const transaction = db.transaction(STORE, mode);
        const store = transaction.objectStore(STORE);
        let request;
        try {
          request = action(store);
        } catch (error) {
          db.close();
          reject(error);
          return;
        }
        transaction.oncomplete = () => {
          db.close();
          resolve(request ? request.result : undefined);
        };
        transaction.onerror = () => {
          db.close();
          reject(transaction.error || new Error("indexedDB transaction failed"));
        };
      }));
    }

    return {
      list() {
        return run("readonly", (store) => store.getAll()).then((rows) => {
          const items = Array.isArray(rows) ? rows : [];
          return items.sort((left, right) => String(right.updated_at).localeCompare(String(left.updated_at)));
        });
      },
      get(id) {
        return run("readonly", (store) => store.get(id));
      },
      save(conversation) {
        const stored = sanitizeConversation(conversation);
        return run("readwrite", (store) => store.put(stored)).then(() => stored);
      },
      delete(id) {
        return run("readwrite", (store) => store.delete(id)).then(() => undefined);
      },
      clear() {
        return run("readwrite", (store) => store.clear()).then(() => undefined);
      },
    };
  }

  return {
    conversationTitle,
    sanitizeCitation,
    sanitizeConversation,
    createConversationStore,
  };
});

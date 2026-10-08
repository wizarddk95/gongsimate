"""Browser conversation records keep public citation fields only."""

from __future__ import annotations

import subprocess

from disclosure_agent.agent.catalog import repo_root

_STORE = repo_root() / "src" / "disclosure_agent" / "agent" / "static" / "conversations.js"


def test_title_truncates_a_question_without_a_model_call() -> None:
    result = _run(
        """
        const store = require(process.argv[1]);
        const title = store.conversationTitle("이수페타시스의 주요 사업은 무엇인가?");
        if (title !== "이수페타시스의 주요 사업") {
          throw new Error(title);
        }
        """
    )
    assert result == ""


def test_saved_conversation_drops_internal_metadata_and_roundtrips() -> None:
    result = _run(
        """
        const store = require(process.argv[1]);
        const saved = store.sanitizeConversation({
          id: "c1",
          title: "",
          created_at: "2026-10-04T01:00:00.000Z",
          updated_at: "2026-10-04T01:05:00.000Z",
          prompt: "hidden",
          messages: [
            {
              role: "user",
              text: "이수페타시스의 주요 사업은 무엇인가?",
              created_at: "2026-10-04T01:00:00.000Z"
            },
            {
              role: "assistant",
              text: "PCB를 생산합니다. [1]",
              created_at: "2026-10-04T01:05:00.000Z",
              kind: "grounded",
              label: "공시 원문 기반",
              retry: true,
              limitations: [],
              citations: [{
                citation_id: "1",
                evidence_id: "ev-1",
                company_name: "이수페타시스",
                report_name: "사업보고서 (2023.12)",
                receipt_no: "20240321001149",
                text: "검색용 문장",
                source_sha256: "abc",
                retrieval_unit_id: "ru-secret",
                corp_code: "00107613",
                score: 0.88,
                excerpt: "검색용",
                blocks: [{
                  type: "table",
                  row_count: 1,
                  col_count: 1,
                  source_sha256: "nope",
                  cells: [{
                    start_row: 0,
                    start_col: 0,
                    rowspan: 1,
                    colspan: 1,
                    text: "사업부문",
                    header: true
                  }]
                }],
                source_locator: [{
                  element_id: "element-1",
                  kind: "table",
                  element_source_locator: {
                    source_sha256: "abc",
                    member_name: "20240321001149.xml",
                    path: [{tag: "TABLE", index: 1}]
                  }
                }]
              }]
            }
          ]
        });
        const blob = JSON.stringify(saved);
        const banned = [
          "source_sha256", "retrieval_unit_id", "corp_code", "score", "prompt", "excerpt"
        ];
        for (const key of banned) {
          if (blob.includes(key)) throw new Error("leaked " + key);
        }
        const round = JSON.parse(blob);
        if (round.id !== "c1") throw new Error("id");
        if (round.title !== "이수페타시스의 주요 사업") throw new Error(round.title);
        const citation = round.messages[1].citations[0];
        if (citation.citation_id !== "1" || citation.evidence_id !== "ev-1") throw new Error("ids");
        if (citation.receipt_no !== "20240321001149") throw new Error("receipt");
        if (round.messages[1].retry !== true) throw new Error("retry");
        if (citation.blocks[0].cells[0].text !== "사업부문") throw new Error("cell");
        if (citation.blocks[0].source_sha256) throw new Error("block hash");
        const member = citation.source_locator[0].element_source_locator.member_name;
        if (member !== "20240321001149.xml") throw new Error("locator");
        const api = store.createConversationStore();
        for (const name of ["list", "get", "save", "delete", "clear"]) {
          if (typeof api[name] !== "function") throw new Error(name);
        }
        """
    )
    assert result == ""


def _run(script: str) -> str:
    completed = subprocess.run(
        ["node", "-e", script, str(_STORE)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr or completed.stdout)
    return completed.stdout.strip()

"""Evidence-bound answers.

``EvidenceBoundAnswerGenerator`` is the deterministic fallback. It quotes
retrieval text and does not add facts that are absent from that text. The
production path uses the OpenAI generator when ``OPENAI_API_KEY`` is set.
Neither generator may mint evidence ids. ``CitationValidator`` rejects
anything outside the current run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from disclosure_agent.agent.citation import Citation
from disclosure_agent.agent.evidence import Evidence, visible_excerpt
from disclosure_agent.agent.settings import CITED_EVIDENCE_LIMIT

_MARKER_RE = re.compile(r"\[(\d+)\]")


@dataclass
class GeneratedAnswer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    abstain: bool = False
    generator_type: str = "deterministic_fallback"
    usage: dict[str, int] | None = None


class EvidenceBoundAnswerGenerator:
    generator_type = "deterministic_fallback"
    uses_llm = False

    def generate(
        self,
        question: str,
        evidence: list[Evidence],
        *,
        company_name: str = "",
        fiscal_year: int | None = None,
        request_time: object = None,
        corpus_as_of: object = None,
    ) -> GeneratedAnswer:
        del question, company_name, fiscal_year, request_time, corpus_as_of
        if not evidence:
            return GeneratedAnswer(
                text=(
                    "해당하는 공시 원문을 찾지 못했습니다. "
                    "확인되지 않은 수치나 원인은 답하지 않습니다."
                ),
                limitations=["검색 결과가 비어 있습니다."],
                abstain=True,
            )
        chosen = evidence[:CITED_EVIDENCE_LIMIT]
        sentences: list[str] = []
        citations: list[Citation] = []
        truncated = False
        for item in chosen:
            passage = visible_excerpt(item.text or item.excerpt)
            if not passage:
                continue
            if passage.endswith("…"):
                truncated = True
            index = len(citations) + 1
            sentences.append(f"{passage} [{index}]")
            citations.append(Citation(citation_id=str(index), evidence_id=item.evidence_id))
        limitations = []
        if truncated:
            limitations.append("발췌는 검색된 구간의 앞부분입니다.")
        if sentences:
            text = "공시에서 확인한 내용입니다. " + " ".join(sentences)
            return GeneratedAnswer(text=text, citations=citations, limitations=limitations)
        for item in chosen:
            index = len(citations) + 1
            citations.append(Citation(citation_id=str(index), evidence_id=item.evidence_id))
            sentences.append(f"[{index}]")
        text = (
            "검색된 문장을 답변으로 옮기지 못했습니다. "
            "번호로 원문을 확인해 주세요. " + " ".join(sentences)
        )
        limitations.append("사용자에게 보일 문장을 분리하지 못했습니다.")
        return GeneratedAnswer(text=text, citations=citations, limitations=limitations)


def company_followup(company_name: str) -> GeneratedAnswer:
    return GeneratedAnswer(
        text=(
            f"{company_name}로 확인했습니다. "
            "매출, 사업, 위험처럼 알고 싶은 내용을 질문해 주세요."
        ),
        generator_type="direct",
    )


def direct_answer(message: str) -> GeneratedAnswer:
    folded = message.strip().casefold()
    if "rag" in folded:
        return GeneratedAnswer(
            text=(
                "그 질문은 특정 기업 공시를 읽지 않습니다. "
                "기업 이름을 포함한 질문을 입력해 주세요."
            ),
            generator_type="direct",
        )
    if folded.startswith("안녕") or folded in {"하이", "hello", "hi"}:
        return GeneratedAnswer(
            text=(
                "안녕하세요. 기업 이름을 포함한 질문을 주시면 "
                "공시 원문을 찾아 답하겠습니다."
            ),
            generator_type="direct",
        )
    return GeneratedAnswer(
        text=(
            "기업 이름을 포함한 공시 질문을 입력해 주세요. "
            "예: 이수페타시스의 주요 사업은 무엇인가요?"
        ),
        generator_type="direct",
    )


def answer_markers_match(answer: GeneratedAnswer) -> bool:
    markers = _MARKER_RE.findall(answer.text)
    return markers == [citation.citation_id for citation in answer.citations]


def answer_has_markers(answer: GeneratedAnswer) -> bool:
    return _MARKER_RE.search(answer.text) is not None


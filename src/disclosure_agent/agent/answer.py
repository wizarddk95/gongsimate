"""Evidence-bound answers.

A future chat model can replace ``EvidenceBoundAnswerGenerator``. It still
receives only the current run's evidence objects and cannot mint evidence ids:
``CitationValidator`` rejects anything else. This default generator quotes
retrieval text and does not add facts that are absent from that text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from disclosure_agent.agent.citation import Citation
from disclosure_agent.agent.company import mentioned_years
from disclosure_agent.agent.evidence import Evidence
from disclosure_agent.agent.settings import CITED_EVIDENCE_LIMIT

_MARKER_RE = re.compile(r"\[(\d+)\]")
_YEAR_TOKEN_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")


@dataclass
class GeneratedAnswer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    abstain: bool = False


class EvidenceBoundAnswerGenerator:
    def generate(self, question: str, evidence: list[Evidence]) -> GeneratedAnswer:
        if not evidence:
            return GeneratedAnswer(
                text="검색된 공시 근거가 없습니다. 확인되지 않은 수치나 원인은 답하지 않습니다.",
                limitations=["검색 결과가 비어 있습니다."],
                abstain=True,
            )
        missing = _missing_years(question, evidence)
        if missing:
            years = ", ".join(missing)
            return GeneratedAnswer(
                text=(
                    f"검색된 공시에서 {years}년 내용을 확인하지 못했습니다. "
                    "근거가 없는 수치는 답하지 않습니다."
                ),
                limitations=[f"현재 검색 근거에 {years}년이 없습니다."],
                abstain=True,
            )
        chosen = evidence[:CITED_EVIDENCE_LIMIT]
        sentences: list[str] = []
        citations: list[Citation] = []
        truncated = False
        for index, item in enumerate(chosen, start=1):
            excerpt = item.excerpt.strip()
            if excerpt.endswith("…"):
                truncated = True
            sentences.append(f"{excerpt} [{index}]")
            citations.append(Citation(citation_id=str(index), evidence_id=item.evidence_id))
        limitations = []
        if truncated:
            limitations.append("발췌는 검색된 구간의 앞부분입니다.")
        text = "공시에서 확인한 내용입니다. " + " ".join(sentences)
        return GeneratedAnswer(text=text, citations=citations, limitations=limitations)


def company_followup(company_name: str) -> GeneratedAnswer:
    return GeneratedAnswer(
        text=(
            f"{company_name}로 확인했습니다. "
            "매출, 사업, 위험처럼 알고 싶은 내용을 질문해 주세요."
        )
    )


def direct_answer(message: str) -> GeneratedAnswer:
    folded = message.strip().casefold()
    if "rag" in folded:
        return GeneratedAnswer(
            text=(
                "RAG는 검색으로 찾은 문서를 답변에 함께 쓰는 방식입니다. "
                "이 답변은 특정 기업 공시를 검색하지 않았습니다."
            )
        )
    if folded.startswith("안녕") or folded in {"하이", "hello", "hi"}:
        return GeneratedAnswer(
            text=(
                "안녕하세요. 기업 이름을 포함한 질문을 주시면 "
                "공시를 찾아 근거와 함께 답하겠습니다."
            )
        )
    return GeneratedAnswer(
        text=(
            "기업 이름을 포함한 공시 질문을 입력해 주세요. "
            "예: 이수페타시스의 주요 사업은 무엇인가요?"
        ),
    )


def answer_markers_match(answer: GeneratedAnswer) -> bool:
    markers = _MARKER_RE.findall(answer.text)
    return markers == [citation.citation_id for citation in answer.citations]


def _missing_years(question: str, evidence: list[Evidence]) -> list[str]:
    years = list(dict.fromkeys(mentioned_years(question)))
    if not years:
        return []
    found: set[str] = set()
    for item in evidence:
        found.update(_YEAR_TOKEN_RE.findall(item.text))
    return [year for year in years if year not in found]

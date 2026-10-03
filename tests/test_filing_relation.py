"""Correction links stay separate from XML parsing."""

from __future__ import annotations

from datetime import datetime

from disclosure_agent.agent.clock import SEOUL
from disclosure_agent.agent.filing_relation import (
    correction_pointer,
    relate_filings,
    suppressed_originals,
)
from disclosure_agent.agent.temporal import FilingPeriod
from disclosure_agent.agent.tool_args import ResolvedSearch, SearchToolArguments, select_filings

NOTICE = """
<CORRECTION>
<TITLE>정 정 신 고 (보고)</TITLE>
<TABLE><TBODY>
<TR><TD>1. 정정대상 공시서류 :</TD><TD>반기보고서</TD></TR>
<TR><TD>2. 정정대상 공시서류의 최초제출일 :</TD><TD>2026년 08월 14일</TD></TR>
</TBODY></TABLE>
</CORRECTION>
"""


def _period(receipt: str, name: str, received: str, corp: str = "00121941") -> FilingPeriod:
    return FilingPeriod(
        receipt_no=receipt,
        report_name=name,
        report_type=name,
        receipt_date=received,
        period_end_year=2026,
        period_end_month=6,
        remark="",
        corp_code=corp,
    )


def test_first_submission_date_is_inferred_not_confirmed() -> None:
    pointer = correction_pointer(NOTICE)
    assert pointer is not None
    assert pointer.document_name == "반기보고서"
    assert pointer.first_submitted_on == "20260814"
    assert pointer.named_receipt_nos == ()
    original = _period("20260814003187", "반기보고서 (2026.06)", "20260814")
    correction = _period("20260911000415", "[기재정정]반기보고서 (2026.06)", "20260911")
    other = _period("20260318000733", "사업보고서 (2025.12)", "20260318")
    relations = relate_filings((original, correction, other), {"20260911000415": pointer})
    assert relations[0].confidence == "inferred"
    assert relations[0].original_receipt_no == "20260814003187"
    dropped = suppressed_originals(
        (original, correction),
        relations,
        as_of=None,
    )
    assert dropped == frozenset({"20260814003187"})


def test_unverified_relation_does_not_drop_the_original() -> None:
    pointer = correction_pointer("<CORRECTION><P>정정사항</P></CORRECTION>")
    assert pointer is not None
    assert pointer.first_submitted_on is None
    original = _period("20260814003187", "반기보고서 (2026.06)", "20260814")
    correction = _period("20260911000415", "[기재정정]반기보고서 (2026.06)", "20260911")
    relations = relate_filings((original, correction), {"20260911000415": pointer})
    assert relations[0].confidence == "unverified"
    assert suppressed_originals((original, correction), relations, as_of=None) == frozenset()


def test_as_of_before_the_correction_keeps_the_original() -> None:
    pointer = correction_pointer(NOTICE)
    assert pointer is not None
    original = _period("20260814003187", "반기보고서 (2026.06)", "20260814")
    correction = _period("20260911000415", "[기재정정]반기보고서 (2026.06)", "20260911")
    relations = relate_filings((original, correction), {"20260911000415": pointer})
    arguments = SearchToolArguments.model_validate(
        {
            "query": "매출",
            "company_id": "00121941",
            "period": None,
            "filing_date_from": None,
            "filing_date_to": None,
            "report_types": None,
            "as_of": "2026-08-20",
            "top_k": 5,
        }
    )
    resolved = ResolvedSearch(
        arguments=arguments,
        filing_date_from=None,
        filing_date_to=None,
        as_of=datetime(2026, 8, 20, tzinfo=SEOUL),
        top_k=5,
    )
    chosen = select_filings(
        (original, correction),
        resolved,
        corpus_as_of=datetime(2026, 10, 1, tzinfo=SEOUL),
        relations=relations,
    )
    assert [item.receipt_no for item in chosen] == ["20260814003187"]

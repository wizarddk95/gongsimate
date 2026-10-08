"""Corrected G0 evaluation contract. These tests do not call a model or search."""

from __future__ import annotations

from pathlib import Path

import pytest

from disclosure_agent.errors import UsageError
from disclosure_agent.evalset_v2.spec import repo_root
from disclosure_agent.evalset_v2_1.g0_evaluation_contract import (
    SOURCE_RUN_DIR,
    classify_claim,
    write_correction,
)
from disclosure_agent.evalset_v2_1.grounded_generation import GENERATOR_INSTRUCTIONS, prompt_sha256


def test_evidence_scoped_nonconfirmation_is_abstention() -> None:
    row = classify_claim("제공된 근거에서는 확인할 수 없다.")
    assert row["claim_type"] == "abstention"


def test_document_absence_stays_a_factual_claim() -> None:
    row = classify_claim("분기보고서에는 해당 정보가 없다.")
    assert row["claim_type"] == "factual_claim"
    assert row["rule"] == "document_absence_is_factual"


def test_scope_limitation_is_not_an_abstention() -> None:
    row = classify_claim("사업보고서에 기재된 기간을 분기보고서의 기간으로 볼 수는 없습니다.")
    assert row["claim_type"] == "limitation"


def test_mixed_wording_stays_unjudged() -> None:
    text = "제공된 근거에서는 확인할 수 없고, 분기보고서에는 해당 정보가 없다."
    assert classify_claim(text)["claim_type"] == "unjudged"


def test_guidance_sentence_is_meta() -> None:
    assert classify_claim("아래와 같이 정리합니다.")["claim_type"] == "meta"


def test_stored_g0_is_rescored_without_overwriting_it(tmp_path: Path) -> None:
    root = repo_root()
    source = root / "evaluation" / "runs" / SOURCE_RUN_DIR
    before = {
        path.name: path.read_bytes()
        for path in source.iterdir()
        if path.is_file()
    }
    with pytest.raises(UsageError):
        write_correction(root, output_dir=source)
    payload = write_correction(root, output_dir=tmp_path)
    after = {
        path.name: path.read_bytes()
        for path in source.iterdir()
        if path.is_file()
    }
    assert after == before
    summary = payload["summary"]
    assert summary["generation_reused"] == 45
    assert summary["new_model_calls"] == 0
    assert summary["judge_recalled"] is False
    assert summary["claim_types"] == {
        "factual_claim": 67,
        "abstention": 7,
        "limitation": 1,
        "meta": 0,
        "unjudged": 0,
    }
    assert summary["faithfulness"]["numerator"] == 66
    assert summary["faithfulness"]["denominator"] == 67
    assert summary["faithfulness"]["unjudged_claim_types"] == 0
    assert summary["citation_completeness"]["numerator"] == 67
    assert summary["citation_completeness"]["denominator"] == 67
    assert summary["citation_correctness"]["numerator"] == 70
    assert summary["citation_correctness"]["denominator"] == 72
    assert summary["citation_correctness"]["excluded_pairs"] == 1
    assert summary["requirement_coverage"]["numerator"] == 51
    assert summary["requirement_coverage"]["denominator"] == 58
    assert summary["success_requirements"] == 50
    assert summary["retrieval_limited_requirements"] == 6
    assert summary["generation_limited_requirements"] == 2
    assert summary["abstention_appropriateness"]["appropriate"] == 6
    assert summary["abstention_appropriateness"]["unjudged"] == 1
    assert payload["manifest"]["generator_prompt_changed"] is False
    assert payload["manifest"]["generator_prompt_sha256"] == prompt_sha256(GENERATOR_INSTRUCTIONS)
    assert payload["manifest"]["retrieval_reexecuted"] is False
    assert payload["manifest"]["test_questions_generated"] == 0

    by_requirement = {
        (result["query_id"], row["requirement_id"]): row
        for result in payload["results"]
        for row in result["requirements"]
    }
    capital = by_requirement[("V2-00126380-finance_table-03", "R1")]
    assert capital["raw_coverage"] == "abstained"
    assert capital["coverage"] == "covered"
    assert capital["primary"] == "success"
    assert capital["support"] == "supported"
    service = by_requirement[("V2-00989619-finance_table-02", "R1")]
    assert service["coverage"] == "abstained"
    assert service["primary"] == "generation_limited"
    assert "over_abstention" in service["flags"]
    assert "citation_missing" not in service["flags"]

    excluded = payload["deltas"]["excluded_citation_pairs"]
    assert excluded == [
        {
            "query_id": "V2-00989619-finance_table-02",
            "claim_id": "K3",
            "claim_type": "limitation",
            "citation_id": "C4",
            "verdict": "supports",
            "reason": "pair verdict kept; claim is outside the factual correctness denominator",
        }
    ]
    names = {
        "manifest.json",
        "evaluation_contract.md",
        "claim_classifications.jsonl",
        "results.jsonl",
        "summary.json",
        "metric_deltas.json",
        "report.md",
    }
    assert names <= {path.name for path in tmp_path.iterdir()}

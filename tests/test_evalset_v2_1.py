"""v2.1 split, retrieval-unit qrels, and metric definitions."""

from __future__ import annotations

import json

from disclosure_agent.chunking import canonical_dumps, sha256_text
from disclosure_agent.evalset_v2.spec import repo_root
from disclosure_agent.evalset_v2_1.freeze import PARENT_DATASET_SHA256, freeze_dataset
from disclosure_agent.evalset_v2_1.metrics import (
    aggregate_headline,
    gain,
    qrel_state,
    score_question,
)
from disclosure_agent.evalset_v2_1.qrels import build_unit_qrels
from disclosure_agent.evalset_v2_1.split import assign_fully_answerable_splits

V2_QREL_COUNT = 5751


def _fully() -> tuple[list[dict], list[dict]]:
    root = repo_root()
    questions = json.loads(
        (
            root / "evaluation/datasets/multi_company_retrieval_v2/questions_frozen.json"
        ).read_text(encoding="utf-8")
    )["questions"]
    requirements = json.loads(
        (root / "evaluation/datasets/multi_company_retrieval_v2/requirements.json").read_text(
            encoding="utf-8"
        )
    )["requirements"]
    fully = [row for row in questions if row["answerability"] == "fully_answerable"]
    return fully, requirements


def _assigned() -> dict[str, str]:
    questions, requirements = _fully()
    return assign_fully_answerable_splits(questions, requirements)


def test_split_is_deterministic_and_ignores_retrieval_fields() -> None:
    questions, requirements = _fully()
    first = assign_fully_answerable_splits(questions, requirements)
    second = assign_fully_answerable_splits(list(reversed(questions)), requirements)
    assert first == second
    noisy = []
    flipped = []
    for question in questions:
        noisy.append(
            {
                **question,
                "dense_rank": 1,
                "dense_score": 0.99,
                "bm25_rank": 3,
                "bm25_score": 12.0,
                "hit": False,
                "preferred_hit": True,
            }
        )
        flipped.append(
            {**question, "split": "dev" if question["split"] == "test" else "test"}
        )
    assert assign_fully_answerable_splits(noisy, requirements) == first
    assert assign_fully_answerable_splits(flipped, requirements) == first


def test_wording_and_query_type_are_in_both_splits() -> None:
    questions, _requirements = _fully()
    assigned = _assigned()
    assert sum(split == "test" for split in assigned.values()) == 55
    assert sum(split == "dev" for split in assigned.values()) == 45
    by_company: dict[str, list[str]] = {}
    wording: dict[str, list[str]] = {}
    query_type: dict[str, list[str]] = {}
    for question in questions:
        split = assigned[question["query_id"]]
        by_company.setdefault(question["corp_code"], []).append(split)
        wording.setdefault(question["wording_style"], []).append(split)
        query_type.setdefault(question["query_type"], []).append(split)
    for splits in by_company.values():
        assert splits.count("test") == 11
        assert splits.count("dev") == 9
    assert wording["original_term"].count("test") == 25
    assert wording["layperson"].count("test") == 25
    assert wording["paraphrase"].count("test") == 5
    assert wording["paraphrase"].count("dev") == 5
    for splits in query_type.values():
        assert splits.count("test") > 0
        assert splits.count("dev") > 0


def test_qrel_is_keyed_by_retrieval_unit_not_copied_elements() -> None:
    questions = [
        {
            "query_id": "Q",
            "seed_unit_ids": ["unit"],
            "answerability": "fully_answerable",
            "split": "test",
        }
    ]
    judgments = [
        {
            "query_id": "Q",
            "retrieval_unit_id": "unit",
            "element_ids": ["e1", "e2", "e3"],
            "document_name": "사업보고서",
            "filing_period": "2025.12",
            "final_source": "agreement",
            "adjudicated": False,
            "models": {"judge_a": "a", "judge_b": "b", "adjudicator": "c"},
            "final": {
                "requirement_judgments": [
                    {"requirement_id": "R1", "relevance": 2},
                    {"requirement_id": "R2", "relevance": 0},
                ]
            },
        }
    ]
    qrels, provenance = build_unit_qrels(judgments, questions)
    assert len(qrels) == 2
    assert {row["requirement_id"] for row in qrels} == {"R1", "R2"}
    assert all("element_id" not in row for row in qrels)
    assert all(row["retrieval_unit_id"] == "unit" for row in qrels)
    assert all(row["judgment_status"] == "judged" for row in qrels)
    assert provenance == [
        {
            "query_id": "Q",
            "retrieval_unit_id": "unit",
            "source_element_ids": ["e1", "e2", "e3"],
        }
    ]
    preferred = [row for row in qrels if row["preferred"]]
    assert [row["requirement_id"] for row in preferred] == ["R1"]


def test_unjudged_is_not_stored_as_irrelevant() -> None:
    assert qrel_state(None) == "unjudged"
    assert qrel_state(0) == "irrelevant"
    questions = [{"query_id": "Q", "seed_unit_ids": ["seed"]}]
    judgments = [
        _judgment("Q", "seed", ["e"], 0),
        {
            "query_id": "Q",
            "retrieval_unit_id": "missing",
            "element_ids": ["x"],
            "final": None,
        },
    ]
    qrels, _provenance = build_unit_qrels(judgments, questions)
    assert [row["retrieval_unit_id"] for row in qrels] == ["seed"]
    assert qrels[0]["relevance"] == 0
    hits = [
        {"rank": 1, "retrieval_unit_id": "missing", "source_element_ids": ["e"]},
        {"rank": 2, "retrieval_unit_id": "seed", "source_element_ids": ["e"]},
    ]
    scored = score_question(
        {"query_id": "Q", "answerability": "fully_answerable", "split": "test"},
        qrels,
        [{"query_id": "Q", "requirement_id": "R1", "answerability": "fully_answerable"}],
        hits,
    )
    assert scored["hits"][0]["relevance"] is None
    assert scored["hits"][0]["judged"] is False
    assert scored["at_k"]["1"]["hit"] is False
    assert scored["hits"][1]["relevance"] == 0
    assert scored["mrr"] == 0.0


def test_shared_element_does_not_copy_relevance_or_inflate_recall() -> None:
    questions = [{"query_id": "Q", "seed_unit_ids": ["relevant"]}]
    judgments = [
        _judgment("Q", "relevant", ["shared", "only-a"], 2),
        _judgment("Q", "irrelevant", ["shared", "only-b"], 0),
    ]
    qrels, provenance = build_unit_qrels(judgments, questions)
    assert len(qrels) == 2
    assert len(provenance) == 2
    hits = [
        {"rank": 1, "retrieval_unit_id": "irrelevant", "source_element_ids": ["shared", "only-b"]}
    ]
    scored = score_question(
        {"query_id": "Q"},
        qrels,
        [{"query_id": "Q", "requirement_id": "R1", "answerability": "fully_answerable"}],
        hits,
    )
    assert scored["at_k"]["1"]["hit"] is False
    assert scored["recall@10"] == {"numerator": 0, "denominator": 1, "value": 0.0}
    assert scored["positive_qrel_count"] == 1
    assert scored["at_k"]["10"]["ndcg_condensed"] < 1


def test_macro_and_micro_recall_and_requirements() -> None:
    requirements = [
        {"query_id": "Q1", "requirement_id": "R1", "answerability": "fully_answerable"},
        {"query_id": "Q1", "requirement_id": "R2", "answerability": "fully_answerable"},
        {"query_id": "Q1", "requirement_id": "R3", "answerability": "fully_answerable"},
        {"query_id": "Q2", "requirement_id": "R1", "answerability": "fully_answerable"},
    ]
    q1 = [
        _qrel("Q1", "R1", "a", 2, preferred=True),
        _qrel("Q1", "R2", "b", 1, preferred=False),
        _qrel("Q1", "R3", "c", 2, preferred=False),
    ]
    q2_qrels = [_qrel("Q2", "R1", f"u{index}", 1, preferred=index == 0) for index in range(10)]
    questions = [
        {"query_id": "Q1", "answerability": "fully_answerable", "split": "test"},
        {"query_id": "Q2", "answerability": "fully_answerable", "split": "test"},
    ]
    first = score_question(
        questions[0],
        q1,
        requirements,
        [
            {"rank": 1, "retrieval_unit_id": "a", "source_element_ids": ["e1", "e2", "e3"]},
            {"rank": 2, "retrieval_unit_id": "b", "source_element_ids": ["e4"]},
        ],
    )
    second = score_question(
        questions[1],
        q2_qrels,
        requirements,
        [{"rank": 1, "retrieval_unit_id": "u0", "source_element_ids": ["x", "y", "z"]}],
    )
    assert first["recall@10"]["value"] == 2 / 3
    assert first["requirement_covered@10"] == 2
    assert first["requirement_total"] == 3
    assert first["all_requirements_covered@10"] is False
    assert first["at_k"]["10"]["hit"] is True
    assert first["at_k"]["1"]["preferred_hit"] is True
    assert second["recall@10"]["value"] == 0.1
    block = aggregate_headline([first, second])
    assert block["recall@10_macro"]["value"] == (2 / 3 + 0.1) / 2
    assert block["recall@10_micro"] == {"numerator": 3, "denominator": 13, "value": 3 / 13}
    assert block["requirement_recall@10"] == {"numerator": 3, "denominator": 4, "value": 0.75}
    assert block["all_requirements_covered@10"]["numerator"] == 1


def test_preferred_fallback_is_not_required_for_relevance() -> None:
    questions = [{"query_id": "Q", "seed_unit_ids": ["seed"]}]
    judgments = [
        _judgment("Q", "seed", ["s"], 0, document_name="사업보고서"),
        _judgment("Q", "other", ["o"], 1, document_name="반기보고서", filing_period="2024.06"),
        _judgment("Q", "annual", ["a"], 2, document_name="사업보고서", filing_period="2023.12"),
    ]
    qrels, _provenance = build_unit_qrels(judgments, questions)
    preferred = {row["retrieval_unit_id"] for row in qrels if row["preferred"]}
    assert preferred == {"annual"}
    scored = score_question(
        {"query_id": "Q"},
        qrels,
        [{"query_id": "Q", "requirement_id": "R1", "answerability": "fully_answerable"}],
        [{"rank": 1, "retrieval_unit_id": "other", "source_element_ids": ["o"]}],
    )
    assert scored["at_k"]["1"]["hit"] is True
    assert scored["at_k"]["1"]["preferred_hit"] is False


def test_ndcg_known_examples() -> None:
    assert gain(2) == 3
    assert gain(1) == 1
    assert gain(0) == 0
    requirements = [
        {"query_id": "Q", "requirement_id": "R1", "answerability": "fully_answerable"}
    ]
    question = {"query_id": "Q"}
    judged = [
        _qrel("Q", "R1", "a", 2, preferred=True),
        _qrel("Q", "R1", "b", 1, preferred=False),
        _qrel("Q", "R1", "c", 0, preferred=False),
    ]
    fully_judged = score_question(
        question,
        judged,
        requirements,
        [_hit(1, "a"), _hit(2, "c"), _hit(3, "b")],
    )
    assert fully_judged["at_k"]["1"]["judged_coverage"]["value"] == 1
    assert (
        fully_judged["at_k"]["5"]["ndcg_condensed"]
        == fully_judged["at_k"]["5"]["ndcg_unjudged_as_zero"]
    )
    assert fully_judged["at_k"]["5"]["ndcg_condensed"] <= 1

    compressed = score_question(
        question,
        [_qrel("Q", "R1", "a", 2, preferred=True), _qrel("Q", "R1", "b", 1, preferred=False)],
        requirements,
        [_hit(1, "a"), _hit(2, "missing"), _hit(3, "b")],
    )
    assert compressed["at_k"]["5"]["unjudged_count"] == 1
    assert compressed["at_k"]["5"]["ndcg_condensed"] == 1
    assert compressed["at_k"]["5"]["ndcg_unjudged_as_zero"] < 1

    outside = score_question(
        question,
        [
            _qrel("Q", "R1", "a", 2, preferred=True),
            _qrel("Q", "R1", "c", 0, preferred=False),
            _qrel("Q", "R1", "d", 2, preferred=False),
        ],
        requirements,
        [_hit(1, "a"), _hit(2, "c")],
    )
    assert outside["at_k"]["5"]["ndcg_condensed"] < 1

    duplicated = score_question(
        question,
        [
            _qrel("Q", "R1", "a", 2, preferred=True),
            _qrel("Q", "R1", "b", 1, preferred=False),
            _qrel("Q", "R1", "d", 2, preferred=False),
        ],
        requirements,
        [_hit(1, "a", ["e1", "e2", "e3"]), _hit(2, "a", ["e1", "e2", "e3"])],
    )
    assert duplicated["recall@10"]["denominator"] == 3
    assert duplicated["recall@10"]["numerator"] == 1
    assert duplicated["at_k"]["5"]["ndcg_condensed"] <= 1
    assert (
        duplicated["at_k"]["5"]["ndcg_condensed"]
        == duplicated["at_k"]["5"]["ndcg_unjudged_as_zero"]
    )


def test_dataset_hash_is_deterministic_and_v2_stays_put() -> None:
    root = repo_root()
    v2_freeze = json.loads(
        (root / "evaluation/datasets/multi_company_retrieval_v2/freeze.json").read_text(
            encoding="utf-8"
        )
    )
    v2_qrels = json.loads(
        (root / "evaluation/datasets/multi_company_retrieval_v2/qrels.json").read_text(
            encoding="utf-8"
        )
    )
    assert v2_freeze["dataset_sha256"] == PARENT_DATASET_SHA256
    assert v2_freeze["qrel_count"] == V2_QREL_COUNT
    assert "element_id" in v2_qrels["qrels"][0]
    first = freeze_dataset(root)
    second = freeze_dataset(root)
    assert first["dataset_sha256"] == second["dataset_sha256"]
    assert first["new_judge_calls"] == 0
    assert first["judgments_reused"] == 2414
    again = json.loads(
        (root / "evaluation/datasets/multi_company_retrieval_v2/freeze.json").read_text(
            encoding="utf-8"
        )
    )
    assert again["dataset_sha256"] == PARENT_DATASET_SHA256
    body = {"qrels": [{"retrieval_unit_id": "u", "relevance": 2}]}
    swapped = {"qrels": [{"relevance": 2, "retrieval_unit_id": "u"}]}
    assert sha256_text(canonical_dumps(body)) == sha256_text(canonical_dumps(swapped))
    changed = {"qrels": [{"retrieval_unit_id": "u", "relevance": 0}]}
    assert sha256_text(canonical_dumps(body)) != sha256_text(canonical_dumps(changed))
    diff = json.loads(
        (root / "evaluation/datasets/multi_company_retrieval_v2_1/dataset_diff.json").read_text(
            encoding="utf-8"
        )
    )
    assert diff["question_text_changed"] == 0
    assert diff["questions_added"] == 0
    assert diff["questions_removed"] == 0
    assert diff["new_judgments"] == 0
    assert diff["llm_api_calls"] == 0
    assert diff["v2_test_positive_elements"] == 502
    assert diff["v2_test_positive_units"] < diff["v2_test_positive_elements"]
    frozen_questions = json.loads(
        (root / "evaluation/datasets/multi_company_retrieval_v2_1/questions.json").read_text(
            encoding="utf-8"
        )
    )["questions"]
    fully = [row for row in frozen_questions if row["answerability"] == "fully_answerable"]
    requirements = json.loads(
        (root / "evaluation/datasets/multi_company_retrieval_v2_1/requirements.json").read_text(
            encoding="utf-8"
        )
    )["requirements"]
    expected = assign_fully_answerable_splits(fully, requirements)
    assert {row["query_id"]: row["split"] for row in fully} == expected


def _judgment(
    query_id: str,
    unit_id: str,
    element_ids: list[str],
    relevance: int,
    *,
    document_name: str = "사업보고서",
    filing_period: str = "2025.12",
) -> dict:
    return {
        "query_id": query_id,
        "retrieval_unit_id": unit_id,
        "element_ids": element_ids,
        "document_name": document_name,
        "filing_period": filing_period,
        "final_source": "agreement",
        "adjudicated": False,
        "models": {},
        "final": {"requirement_judgments": [{"requirement_id": "R1", "relevance": relevance}]},
    }


def _qrel(
    query_id: str,
    requirement_id: str,
    unit_id: str,
    relevance: int,
    *,
    preferred: bool,
) -> dict:
    return {
        "query_id": query_id,
        "requirement_id": requirement_id,
        "retrieval_unit_id": unit_id,
        "relevance": relevance,
        "preferred": preferred,
        "judgment_status": "judged",
    }


def _hit(rank: int, unit_id: str, element_ids: list[str] | None = None) -> dict:
    return {
        "rank": rank,
        "retrieval_unit_id": unit_id,
        "source_element_ids": element_ids or [unit_id],
    }


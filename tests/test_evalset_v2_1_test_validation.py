"""One-shot TEST reading of the frozen H10 policy. These tests do not search."""

from __future__ import annotations

from pathlib import Path

import pytest

from disclosure_agent.errors import UsageError
from disclosure_agent.evalset_v2_1.asymmetric_fusion import (
    RANKING_DROP_LIMIT,
    RULE_ID,
    bm25_primary_dense_head,
)
from disclosure_agent.evalset_v2_1.test_validation import (
    FROZEN_COMMIT,
    RUN_DIR_NAME,
    final_retrieval_decision,
    hit_transition,
    judge_generalization,
    requirement_pair,
    run_test_validation,
    select_test_questions,
)


def _verdict(**overrides: object) -> dict:
    payload = {
        "bm25_hit10": 40,
        "h10_hit10": 41,
        "bm25_only": 0,
        "h10_only": 1,
        "requirement_gained": 1,
        "requirement_lost": 0,
        "recall5_drop": 0.0,
        "recall10_drop": -0.01,
        "mrr_drop": -0.002,
        "ndcg_drop": -0.004,
    }
    payload.update(overrides)
    return judge_generalization(**payload)  # type: ignore[arg-type]


def test_support_does_not_reuse_the_dev_numerator_floor() -> None:
    verdict = _verdict(bm25_hit10=9, h10_hit10=10)
    assert verdict["label"] == "Generalization Support"
    assert verdict["reason"] == "bm25_preserved_and_semantic_recovery"
    assert verdict["dev_hit10_floors_applied"] is False
    assert verdict["ranking_drop_limit"] == RANKING_DROP_LIMIT


def test_identical_lists_are_mixed() -> None:
    verdict = _verdict(
        bm25_hit10=40,
        h10_hit10=40,
        bm25_only=0,
        h10_only=0,
        requirement_gained=0,
        requirement_lost=0,
        recall10_drop=0.0,
        mrr_drop=0.0,
        ndcg_drop=0.0,
    )
    assert verdict["label"] == "Neutral / Mixed"
    assert verdict["reason"] == "no_additional_recovery"


def test_hit_gain_and_loss_offset_is_mixed() -> None:
    verdict = _verdict(bm25_hit10=40, h10_hit10=40, bm25_only=1, h10_only=1)
    assert verdict["label"] == "Neutral / Mixed"
    assert verdict["reason"] == "hit10_gains_and_losses"


def test_requirement_offset_blocks_support() -> None:
    verdict = _verdict(requirement_gained=1, requirement_lost=1)
    assert verdict["label"] == "Neutral / Mixed"
    assert "requirement_gains_and_losses_offset" in verdict["mixed_reasons"]


def test_any_hit10_drop_is_failure() -> None:
    verdict = _verdict(bm25_hit10=10, h10_hit10=9, bm25_only=1, h10_only=0)
    assert verdict["label"] == "Generalization Failure"
    assert verdict["reason"] == "hit10_regression"


def test_ranking_drop_above_the_frozen_limit_is_failure() -> None:
    verdict = _verdict(ndcg_drop=RANKING_DROP_LIMIT + 0.001)
    assert verdict["label"] == "Generalization Failure"
    assert verdict["reason"] == "ranking_regression"


def test_ranking_drop_at_the_frozen_limit_still_supports() -> None:
    verdict = _verdict(ndcg_drop=RANKING_DROP_LIMIT)
    assert verdict["label"] == "Generalization Support"


def test_requirement_net_loss_is_failure() -> None:
    verdict = _verdict(requirement_gained=0, requirement_lost=1)
    assert verdict["label"] == "Generalization Failure"
    assert verdict["reason"] == "requirement_net_loss"


def test_requirement_net_gain_can_support() -> None:
    verdict = _verdict(requirement_gained=2, requirement_lost=1)
    assert verdict["label"] == "Generalization Support"


def test_non_reproduction_selects_bm25_without_retuning() -> None:
    mixed = final_retrieval_decision("Neutral / Mixed")
    failed = final_retrieval_decision("Generalization Failure")
    supported = final_retrieval_decision("Generalization Support")
    assert mixed["selected_configuration"] == "bm25"
    assert failed["selected_configuration"] == "bm25"
    assert supported["selected_configuration"] == RULE_ID
    assert mixed["retune"] is False
    assert failed["retune"] is False
    assert supported["retune"] is False
    assert "BM25 baseline을 최종 선택했다" in mixed["statement"]
    assert "semantic miss를 추가 복구했다" in supported["statement"]


def test_transition_names_match_the_registered_partition() -> None:
    assert hit_transition(True, True) == "both_hit"
    assert hit_transition(True, False) == "bm25_only"
    assert hit_transition(False, True) == "h10_only"
    assert hit_transition(False, False) == "both_miss"
    assert requirement_pair(True, True) == "both"
    assert requirement_pair(True, False) == "bm25_only"
    assert requirement_pair(False, True) == "h10_only"
    assert requirement_pair(False, False) == "neither"


def test_population_keeps_test_fully_answerable_only() -> None:
    questions = [
        {"query_id": "t", "answerability": "fully_answerable", "split": "test"},
        {"query_id": "d", "answerability": "fully_answerable", "split": "dev"},
        {"query_id": "p", "answerability": "partially_answerable", "split": "test"},
        {"query_id": "u", "answerability": "unanswerable", "split": "test"},
    ]
    assert [row["query_id"] for row in select_test_questions(questions)] == ["t"]


def test_frozen_commit_and_rule_stay_the_dev_selection() -> None:
    assert FROZEN_COMMIT.startswith("f911283")
    assert RULE_ID == "bm25_primary_dense_head"
    assert bm25_primary_dense_head.__module__.endswith("asymmetric_fusion")


def test_first_call_searches_once_and_a_lock_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []

    def search(root: Path) -> None:
        calls.append(root)

    monkeypatch.setattr(
        "disclosure_agent.evalset_v2_1.test_validation._search_and_lock",
        search,
    )
    monkeypatch.setattr(
        "disclosure_agent.evalset_v2_1.test_validation.score_stored_test",
        lambda root: {"rescored": True},
    )
    assert run_test_validation(tmp_path) == {"rescored": True}
    assert calls == [tmp_path]

    run_dir = tmp_path / "evaluation" / "runs" / RUN_DIR_NAME
    run_dir.mkdir(parents=True)
    (run_dir / "execution_lock.json").write_text("{}\n", encoding="utf-8")
    assert run_test_validation(tmp_path) == {"rescored": True}
    assert calls == [tmp_path]


def test_a_started_search_refuses_a_second_retrieval(tmp_path: Path) -> None:
    from disclosure_agent.evalset_v2_1.test_validation import _search_and_lock

    run_dir = tmp_path / "evaluation" / "runs" / RUN_DIR_NAME
    run_dir.mkdir(parents=True)
    (run_dir / "search_started.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(UsageError, match="second retrieval"):
        _search_and_lock(tmp_path)

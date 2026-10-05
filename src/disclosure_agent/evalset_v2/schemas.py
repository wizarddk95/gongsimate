"""Structured objects for generation, decomposition, and judging."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class GeneratedQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    query_type: str
    difficulty_tags: list[str] = Field(default_factory=list)


class PartialQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    answerable_requirement: str
    unanswerable_requirement: str


class BareQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str


class AtomicRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement_id: str
    text: str
    evidence_labels: list[str] = Field(default_factory=list)


class Decomposition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirements: list[AtomicRequirement]


class RequirementJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement_id: str
    relevance: int


class PassageJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement_judgments: list[RequirementJudgment]
    passage_relevance: int

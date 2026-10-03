"""Agent-run failures. User-facing text stays separate from internal detail."""

from __future__ import annotations

import re

_SECRET_RE = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")


class AgentFlowError(Exception):
    """A controlled run failure. The API maps ``code`` to an SSE error event."""

    code = "internal_error"

    def __init__(self, detail: str = "") -> None:
        super().__init__(detail or self.code)
        self.detail = detail


class RetrievalFailure(AgentFlowError):
    code = "retrieval_failure"


class GenerationFailure(AgentFlowError):
    code = "generation_failure"


class CitationValidationFailure(AgentFlowError):
    code = "citation_validation_failure"


class OpenAIAuthError(AgentFlowError):
    code = "openai_auth_error"


class OpenAIRateLimitError(AgentFlowError):
    code = "openai_rate_limit"


class OpenAITimeoutError(AgentFlowError):
    code = "openai_timeout"


class OpenAIAPIError(AgentFlowError):
    code = "openai_api_error"


class StructuredOutputInvalid(AgentFlowError):
    code = "structured_output_invalid"


class RequestTimeout(AgentFlowError):
    code = "request_timeout"


class LlmLimitExceeded(AgentFlowError):
    code = "llm_limit_exceeded"


class SearchLimitExceeded(AgentFlowError):
    code = "search_limit_exceeded"


USER_MESSAGES: dict[str, str] = {
    "retrieval_failure": "공시를 검색하는 중 문제가 생겼습니다. 잠시 후 다시 시도해 주세요.",
    "generation_failure": "답변을 만들지 못했습니다. 잠시 후 다시 시도해 주세요.",
    "citation_validation_failure": "답변과 근거 연결을 확인하지 못해 답변을 보류했습니다.",
    "openai_auth_error": "답변 모델 인증에 실패했습니다. 설정을 확인해 주세요.",
    "openai_rate_limit": "답변 모델 요청 한도에 도달했습니다. 잠시 후 다시 시도해 주세요.",
    "openai_timeout": "답변 모델 응답 시간이 초과되었습니다.",
    "openai_api_error": "답변 모델을 호출하지 못했습니다. 잠시 후 다시 시도해 주세요.",
    "structured_output_invalid": "답변 형식을 확인하지 못해 답변을 보류했습니다.",
    "request_timeout": "요청 처리 시간이 초과되었습니다.",
    "llm_limit_exceeded": "이번 질문에서 사용할 수 있는 처리 한도에 도달했습니다.",
    "search_limit_exceeded": "이번 질문에서 검색 횟수 한도에 도달했습니다.",
    "internal_error": "요청을 처리하지 못했습니다.",
    "company_ambiguous": "어느 기업을 말씀하시는지 확인해 주세요.",
    "company_not_found": "말씀하신 기업을 공시 기업 목록에서 찾지 못했습니다.",
    "company_unsupported": "기업은 확인했지만, 현재 검색할 수 있는 공시가 없습니다.",
    "retrieval_no_result": "검색된 공시 근거가 없습니다.",
    "abstained": "검색된 공시에서 요청한 내용을 확인하지 못했습니다.",
}


def user_message(code: str) -> str:
    return USER_MESSAGES.get(code, USER_MESSAGES["internal_error"])


def redact_text(text: str, secrets: tuple[str, ...] = ()) -> str:
    """Remove configured secrets and OpenAI-style key material from log text."""
    redacted = text
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return _SECRET_RE.sub("[REDACTED]", redacted)

"""POST /ask as server-sent events. The runtime is injected so tests avoid the corpus."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from disclosure_agent.agent.runtime import AgentRuntime, ClarificationContext, build_default_runtime
from disclosure_agent.agent.search_catalog import public_corpus
from disclosure_agent.agent.sse import format_sse

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_STATIC_FILES = {
    "chat.js": "chat.js",
    "chat.css": "chat.css",
    "conversations.js": "conversations.js",
}

_MAX_MESSAGE_CHARS = 4000


class ClarificationIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prior_message: str | None = None
    selected_company_name: str | None = None


class AskIn(BaseModel):
    """The client sends a question. company_id is not an input and is ignored."""

    model_config = ConfigDict(extra="ignore")

    message: str = Field(min_length=1)
    conversation_id: str | None = None
    clarification: ClarificationIn | None = None


def create_app(runtime: AgentRuntime | None = None) -> FastAPI:
    application = FastAPI(title="disclosure-agent", version="0.1.0")
    application.state.runtime = runtime

    @application.get("/", response_model=None)
    def index() -> FileResponse:
        return FileResponse(
            _STATIC_DIR / "index.html",
            media_type="text/html",
            headers={"Cache-Control": "no-cache"},
        )

    @application.get("/static/{name}", response_model=None)
    def static_file(name: str) -> FileResponse | JSONResponse:
        filename = _STATIC_FILES.get(name)
        if filename is None:
            return JSONResponse(
                status_code=404,
                content={"code": "not_found", "message": "요청한 파일을 찾지 못했습니다."},
            )
        return FileResponse(_STATIC_DIR / filename, headers={"Cache-Control": "no-cache"})

    @application.get("/api/corpus")
    def read_corpus() -> JSONResponse:
        agent = application.state.runtime
        if not isinstance(agent, AgentRuntime):
            return JSONResponse(
                status_code=500,
                content={"code": "internal_error", "message": "에이전트가 준비되지 않았습니다."},
            )
        return JSONResponse(public_corpus(agent.catalog))

    @application.post("/ask", response_model=None)
    def ask(body: AskIn) -> StreamingResponse | JSONResponse:
        message = body.message.strip()
        if not message:
            return JSONResponse(
                status_code=400,
                content={"code": "invalid_request", "message": "질문을 입력해 주세요."},
            )
        if len(message) > _MAX_MESSAGE_CHARS:
            return JSONResponse(
                status_code=400,
                content={"code": "invalid_request", "message": "질문이 너무 깁니다."},
            )
        agent = application.state.runtime
        if not isinstance(agent, AgentRuntime):
            return JSONResponse(
                status_code=500,
                content={"code": "internal_error", "message": "에이전트가 준비되지 않았습니다."},
            )
        clarification = None
        if body.clarification and body.clarification.selected_company_name:
            clarification = ClarificationContext(
                prior_message=body.clarification.prior_message,
                selected_company_name=body.clarification.selected_company_name,
            )

        def events() -> Any:
            for event in agent.stream(
                message,
                conversation_id=body.conversation_id,
                clarification=clarification,
            ):
                yield format_sse(event)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return application


_app: FastAPI | None = None


def get_app() -> FastAPI:
    global _app
    if _app is None:
        _app = create_app(build_default_runtime())
    return _app


def __getattr__(name: str) -> Any:
    if name == "app":
        return get_app()
    raise AttributeError(name)


def main() -> None:
    import uvicorn

    uvicorn.run(get_app(), host="127.0.0.1", port=8000)

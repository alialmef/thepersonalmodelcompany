"""FastAPI router for /v1/agent/*.

  GET  /v1/agent/providers           — list known providers (no auth — public catalog)
  GET  /v1/agent/config              — current user's provider/model (no key returned)
  PUT  /v1/agent/config              — set provider + model + api_key (key encrypted + stored)
  DELETE /v1/agent/config            — clear stored config
  POST /v1/agent/config/validate     — test a key without storing (optional)
  POST /v1/agent/chat                — non-streaming chat
  POST /v1/agent/chat/stream         — streaming chat (SSE: data: <chunk>\\n\\n)

Mounted from pmc/serve/api.py when AuthStore is configured.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from pmc.agent import crypto
from pmc.agent.prompts import TaskKind, compose
from pmc.agent.providers.base import (
    Message,
    ProviderConfig,
    ProviderError,
)
from pmc.agent.providers.registry import (
    KNOWN_PROVIDERS,
    get_provider,
    is_known_provider,
)
from pmc.auth.middleware import AuthSession, require_session


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ProviderInfo(BaseModel):
    id: str
    label: str
    default_models: list[str]
    key_prefix_hint: Optional[str] = None
    console_url: Optional[str] = None


class ProvidersResponse(BaseModel):
    providers: list[ProviderInfo]


class ConfigResponse(BaseModel):
    configured: bool
    provider: Optional[str] = None
    model: Optional[str] = None
    updated_at: Optional[str] = None
    encryption_configured: bool


class SetConfigRequest(BaseModel):
    provider: str = Field(..., min_length=1, max_length=64)
    model: str = Field(..., min_length=1, max_length=128)
    api_key: str = Field(..., min_length=1, max_length=512)


class ValidateKeyRequest(BaseModel):
    provider: str
    api_key: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    system: Optional[str] = None
    max_tokens: int = 4096
    model: Optional[str] = None  # optional override of stored config


class ChatResponse(BaseModel):
    text: str
    model: str
    usage: dict
    finish_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store(request: Request):
    s = getattr(request.app.state, "auth_store", None)
    if s is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="auth backend not configured",
        )
    return s


def _load_config(request: Request, account_id: str) -> ProviderConfig:
    store = _store(request)
    cfg = store.get_provider_config(account_id)
    if not cfg:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "provider_not_configured",
                "message": "Set your provider + model + API key in Settings first.",
            },
        )
    try:
        api_key = crypto.decrypt(cfg["api_key_ciphertext"])
    except crypto.EncryptionNotConfigured as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        ) from e
    return ProviderConfig(
        provider=cfg["provider"], model=cfg["model"], api_key=api_key
    )


def _to_http(e: ProviderError) -> HTTPException:
    code = {
        "auth": status.HTTP_401_UNAUTHORIZED,
        "rate_limit": status.HTTP_429_TOO_MANY_REQUESTS,
        "model": status.HTTP_400_BAD_REQUEST,
        "network": status.HTTP_502_BAD_GATEWAY,
    }.get(e.kind, status.HTTP_502_BAD_GATEWAY)
    return HTTPException(status_code=code, detail={"error": e.kind, "message": str(e)})


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def build_agent_router() -> APIRouter:
    router = APIRouter(prefix="/v1/agent", tags=["agent"])

    @router.get("/providers", response_model=ProvidersResponse)
    def list_providers() -> ProvidersResponse:
        return ProvidersResponse(
            providers=[ProviderInfo(**p) for p in KNOWN_PROVIDERS]
        )

    @router.get("/config", response_model=ConfigResponse)
    def get_config(
        request: Request,
        session: AuthSession = Depends(require_session),
    ) -> ConfigResponse:
        store = _store(request)
        cfg = store.get_provider_config(session.account.id)
        return ConfigResponse(
            configured=cfg is not None,
            provider=cfg["provider"] if cfg else None,
            model=cfg["model"] if cfg else None,
            updated_at=cfg["updated_at"].isoformat() if cfg else None,
            encryption_configured=crypto.is_configured(),
        )

    @router.put("/config", response_model=ConfigResponse)
    async def set_config(
        body: SetConfigRequest,
        request: Request,
        session: AuthSession = Depends(require_session),
    ) -> ConfigResponse:
        if not is_known_provider(body.provider):
            raise HTTPException(
                status_code=400,
                detail=f"unknown provider {body.provider!r}",
            )
        if not crypto.is_configured():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "PMC_KEY_ENCRYPTION_SECRET not set on this deploy — "
                    "refusing to store user API keys"
                ),
            )
        # Probe the key before persisting. Catches bad keys at the
        # Settings UI rather than at first chat.
        provider = get_provider(body.provider)
        if provider is None:
            raise HTTPException(status_code=400, detail="unknown provider")
        ok = await provider.validate_key(api_key=body.api_key)
        if not ok:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="key didn't validate against the provider",
            )

        ct = crypto.encrypt(body.api_key)
        store = _store(request)
        store.set_provider_config(
            session.account.id,
            provider=body.provider,
            model=body.model,
            api_key_ciphertext=ct,
        )
        cfg = store.get_provider_config(session.account.id)
        return ConfigResponse(
            configured=True,
            provider=cfg["provider"],
            model=cfg["model"],
            updated_at=cfg["updated_at"].isoformat(),
            encryption_configured=True,
        )

    @router.delete("/config")
    def clear_config(
        request: Request,
        session: AuthSession = Depends(require_session),
    ) -> dict:
        _store(request).clear_provider_config(session.account.id)
        return {"ok": True}

    @router.post("/config/validate")
    async def validate(body: ValidateKeyRequest) -> dict:
        if not is_known_provider(body.provider):
            raise HTTPException(status_code=400, detail="unknown provider")
        provider = get_provider(body.provider)
        ok = await provider.validate_key(api_key=body.api_key)  # type: ignore[union-attr]
        return {"ok": ok}

    @router.post("/chat", response_model=ChatResponse)
    async def chat(
        body: ChatRequest,
        request: Request,
        session: AuthSession = Depends(require_session),
    ) -> ChatResponse:
        cfg = _load_config(request, session.account.id)
        if body.model:
            cfg = ProviderConfig(
                provider=cfg.provider, model=body.model, api_key=cfg.api_key
            )
        provider = get_provider(cfg.provider)
        if provider is None:
            raise HTTPException(status_code=400, detail="unknown provider in config")
        msgs = [Message(role=m.role, content=m.content) for m in body.messages]
        try:
            resp = await provider.chat(
                msgs,
                config=cfg,
                max_tokens=body.max_tokens,
                system=body.system,
            )
        except ProviderError as e:
            raise _to_http(e) from e
        return ChatResponse(
            text=resp.text,
            model=resp.model,
            usage=resp.usage,
            finish_reason=resp.finish_reason,
        )

    @router.post("/chat/stream")
    async def chat_stream(
        body: ChatRequest,
        request: Request,
        session: AuthSession = Depends(require_session),
    ):
        cfg = _load_config(request, session.account.id)
        if body.model:
            cfg = ProviderConfig(
                provider=cfg.provider, model=body.model, api_key=cfg.api_key
            )
        provider = get_provider(cfg.provider)
        if provider is None:
            raise HTTPException(status_code=400, detail="unknown provider in config")
        msgs = [Message(role=m.role, content=m.content) for m in body.messages]

        async def event_source():
            try:
                async for chunk in provider.stream_chat(  # type: ignore[union-attr]
                    msgs,
                    config=cfg,
                    max_tokens=body.max_tokens,
                    system=body.system,
                ):
                    # SSE framing — caller reads as text/event-stream
                    yield f"data: {chunk}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
            except ProviderError as e:
                yield f"event: error\ndata: {{\"kind\":\"{e.kind}\",\"message\":\"{str(e)[:300]}\"}}\n\n"

        return StreamingResponse(event_source(), media_type="text/event-stream")

    return router

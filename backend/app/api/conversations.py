"""Conversation, chat (SSE streaming), version and export endpoints."""

from __future__ import annotations

import json
import logging
import re
from typing import Literal

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.api.resumes import read_upload
from app.parsing.extract import ResumeParseError
from app.services.chat import ChatError, ChatService
from app.services.export import to_docx, to_markdown
from app.services.versioning import VersionError
from app.storage.base import NotFoundError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/conversations", tags=["conversations"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20000)
    job_description: str | None = Field(default=None, max_length=30000)


def service(request: Request) -> ChatService:
    return request.app.state.chat_service


def _not_found(exc: NotFoundError) -> HTTPException:
    return HTTPException(404, f"Not found: {exc.args[0] if exc.args else ''}")


# ------------------------------------------------------------------ conversations
@router.post("", status_code=201)
async def create_conversation(request: Request) -> dict:
    conv = await service(request).create_conversation()
    return conv.model_dump(mode="json")


@router.get("")
def list_conversations(request: Request, limit: int = Query(30, le=100)) -> list[dict]:
    convs = service(request).repo.list_conversations(limit)
    return [
        {
            "id": c.id,
            "title": c.title,
            "updated_at": c.updated_at.isoformat(),
            "filename": c.filename,
        }
        for c in convs
    ]


@router.get("/{conversation_id}")
async def get_conversation(request: Request, conversation_id: str) -> dict:
    try:
        return await service(request).get_state(conversation_id)
    except NotFoundError as exc:
        raise _not_found(exc) from exc


@router.delete("/{conversation_id}", status_code=204)
def delete_conversation(request: Request, conversation_id: str) -> Response:
    service(request).repo.delete_conversation(conversation_id)
    return Response(status_code=204)


@router.post("/{conversation_id}/resume")
async def upload_resume(
    request: Request, conversation_id: str, file: UploadFile = File(...)
) -> dict:
    data = await read_upload(request, file)
    try:
        return await service(request).upload_resume(
            conversation_id, file.filename or "resume", data
        )
    except ResumeParseError as exc:
        raise HTTPException(422, str(exc)) from exc
    except NotFoundError as exc:
        raise _not_found(exc) from exc


# ---------------------------------------------------------------------------- chat
@router.post("/{conversation_id}/chat")
async def chat(request: Request, conversation_id: str, body: ChatRequest) -> EventSourceResponse:
    svc = service(request)
    try:
        await svc._io(svc.repo.get_conversation, conversation_id)
    except NotFoundError as exc:
        raise _not_found(exc) from exc

    async def events():
        try:
            async for event in svc.chat(conversation_id, body.message, body.job_description):
                yield {"event": event["type"], "data": json.dumps(event, default=str)}
        except ChatError as exc:
            yield {"event": "error", "data": json.dumps({"type": "error", "message": str(exc)})}
        except Exception:
            logger.exception("Chat turn failed")
            message = "Something went wrong while processing that request. Please try again."
            yield {"event": "error", "data": json.dumps({"type": "error", "message": message})}

    return EventSourceResponse(events(), ping=15)


# ------------------------------------------------------------------------ versions
@router.get("/{conversation_id}/versions")
def list_versions(request: Request, conversation_id: str) -> list[dict]:
    try:
        versions = service(request).repo.list_versions(conversation_id)
    except NotFoundError as exc:
        raise _not_found(exc) from exc
    return [{**v.summary(), "created_at": v.created_at.isoformat()} for v in versions]


@router.get("/{conversation_id}/versions/{version_id}")
def get_version(request: Request, conversation_id: str, version_id: str) -> dict:
    try:
        return (
            service(request).repo.get_version(conversation_id, version_id).model_dump(mode="json")
        )
    except NotFoundError as exc:
        raise _not_found(exc) from exc


async def _version_op(
    request: Request, conversation_id: str, op: str, version_id: str | None = None
):
    try:
        return await service(request).version_op(conversation_id, op, version_id)
    except VersionError as exc:
        raise HTTPException(409, str(exc)) from exc
    except NotFoundError as exc:
        raise _not_found(exc) from exc


@router.post("/{conversation_id}/undo")
async def undo(request: Request, conversation_id: str) -> dict:
    return await _version_op(request, conversation_id, "undo")


@router.post("/{conversation_id}/redo")
async def redo(request: Request, conversation_id: str) -> dict:
    return await _version_op(request, conversation_id, "redo")


@router.post("/{conversation_id}/versions/{version_id}/restore")
async def restore(request: Request, conversation_id: str, version_id: str) -> dict:
    return await _version_op(request, conversation_id, "restore", version_id)


@router.get("/{conversation_id}/compare")
def compare(
    request: Request,
    conversation_id: str,
    from_id: str = Query(alias="from"),
    to_id: str = Query(alias="to"),
) -> list[dict]:
    svc = service(request)
    try:
        conv = svc.repo.get_conversation(conversation_id)
        return [c.model_dump() for c in svc.versions.compare(conv, from_id, to_id)]
    except NotFoundError as exc:
        raise _not_found(exc) from exc


@router.get("/{conversation_id}/export")
def export(
    request: Request,
    conversation_id: str,
    format: Literal["docx", "md"] = "docx",
    version_id: str | None = None,
) -> Response:
    svc = service(request)
    try:
        conv = svc.repo.get_conversation(conversation_id)
        vid = version_id or conv.current_version_id
        if not vid:
            raise HTTPException(404, "No resume uploaded yet.")
        version = svc.repo.get_version(conversation_id, vid)
    except NotFoundError as exc:
        raise _not_found(exc) from exc
    stem = re.sub(r"[^\w-]+", "_", (conv.filename or "resume").rsplit(".", 1)[0])
    name = f"{stem}_v{version.number}.{format}"
    if format == "docx":
        body = to_docx(version.resume)
        media = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    else:
        body, media = to_markdown(version.resume), "text/markdown; charset=utf-8"
    return Response(
        body, media_type=media, headers={"Content-Disposition": f'attachment; filename="{name}"'}
    )

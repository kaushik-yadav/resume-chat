"""Stateless resume parsing endpoint (useful for previews and debugging the parser)."""

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.models.resume import Resume
from app.parsing.extract import ResumeParseError
from app.parsing.sectionizer import parse_resume

router = APIRouter(prefix="/resumes", tags=["resumes"])


async def read_upload(request: Request, file: UploadFile) -> bytes:
    limit = request.app.state.settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File is larger than {limit // (1024 * 1024)} MB.")
    return data


@router.post("/parse", response_model=Resume)
async def parse(request: Request, file: UploadFile = File(...)) -> Resume:
    data = await read_upload(request, file)
    try:
        return parse_resume(file.filename or "", data)
    except ResumeParseError as exc:
        raise HTTPException(422, str(exc)) from exc

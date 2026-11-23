"""Firebase Firestore repository (conversation history and resume versions)."""

from __future__ import annotations

import json
import logging
import os

import firebase_admin
from firebase_admin import credentials, firestore
from google.api_core.client_options import ClientOptions
from google.auth.credentials import AnonymousCredentials
from google.auth.credentials import Credentials as GoogleCredentials
from google.cloud.firestore import Client
from google.cloud.firestore_v1.base_query import FieldFilter
from google.cloud.firestore_v1.services.firestore import FirestoreClient
from google.oauth2.credentials import Credentials as OAuthCredentials

from app.config import Settings
from app.models.conversation import ChatMessage, Conversation, ResumeVersion
from app.storage.base import NotFoundError, Repository

logger = logging.getLogger(__name__)

_APP_NAME = "resume-chat"


def _tolerate_whitespace_between_stream_chunks() -> None:
    """Make google-api-core's REST stream parser skip whitespace between top-level values.

    The Firestore emulator ends streamed responses (runQuery, batchGet) with ``]\\n``; the
    parser treats the trailing newline as the start of a new array and raises
    "Can only parse array of JSON objects". Production Firestore is unaffected either way.
    """
    from google.api_core._rest_streaming_base import BaseResponseIterator

    if getattr(BaseResponseIterator, "_resume_chat_patched", False):
        return
    original = BaseResponseIterator._process_chunk

    def process_chunk(self, chunk: str):
        if self._level == 0:
            chunk = chunk.lstrip()
            if not chunk:
                return
        original(self, chunk)

    BaseResponseIterator._process_chunk = process_chunk
    BaseResponseIterator._resume_chat_patched = True


class RestFirestoreClient(Client):
    """Firestore client that speaks HTTP/1.1 REST instead of gRPC.

    Some networks (corporate proxies, certain VPN/firewall drivers on Windows) break gRPC's
    HTTP/2 connections. The REST transport supports every call this app makes (CRUD, queries,
    batched reads); it lacks only real-time listeners, which the backend doesn't use.
    """

    def __init__(self, *, endpoint: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._rest_endpoint = endpoint
        _tolerate_whitespace_between_stream_chunks()

    @property
    def _firestore_api(self):
        if self._firestore_api_internal is None:
            options = (
                ClientOptions(api_endpoint=self._rest_endpoint) if self._rest_endpoint else None
            )
            self._firestore_api_internal = FirestoreClient(
                transport="rest", credentials=self._credentials, client_options=options
            )
        return self._firestore_api_internal


def _admin_credentials(settings: Settings) -> GoogleCredentials:
    try:
        app = firebase_admin.get_app(_APP_NAME)
    except ValueError:
        cred = (
            credentials.Certificate(settings.firebase_credentials)
            if settings.firebase_credentials
            else credentials.ApplicationDefault()
        )
        app = firebase_admin.initialize_app(
            cred, {"projectId": settings.firebase_project_id}, name=_APP_NAME
        )
    return app.credential.get_credential()


def build_client(settings: Settings) -> Client:
    """Create a Firestore client for the emulator or a real Firebase project."""
    project = settings.firebase_project_id
    rest = settings.firestore_transport == "rest"

    if settings.firestore_emulator_host:
        host = settings.firestore_emulator_host
        logger.info("Using Firestore emulator at %s (%s)", host, settings.firestore_transport)
        if rest:
            # "owner" is the emulator's admin token: it bypasses security rules, matching how
            # the Admin SDK behaves against production.
            return RestFirestoreClient(
                endpoint=f"http://{host}", project=project, credentials=OAuthCredentials("owner")
            )
        # The Google client libraries discover the emulator through this env var.
        os.environ["FIRESTORE_EMULATOR_HOST"] = host
        return Client(project=project, credentials=AnonymousCredentials())

    logger.info("Using Firestore project %s (%s)", project, settings.firestore_transport)
    if rest:
        return RestFirestoreClient(project=project, credentials=_admin_credentials(settings))
    try:
        app = firebase_admin.get_app(_APP_NAME)
    except ValueError:
        _admin_credentials(settings)
        app = firebase_admin.get_app(_APP_NAME)
    return firestore.client(app)


# Large nested fields are stored as JSON strings. Firestore never needs to query inside them,
# and converting deeply nested maps to and from Firestore values was the dominant cost of
# loading a conversation (13 messages with traces: ~1.4 s). Documents written before this
# change (nested maps) are still read correctly.
_JSON_FIELDS = {"message": ("trace",), "version": ("resume", "changes")}


def _encode(data: dict, fields: tuple[str, ...]) -> dict:
    for name in fields:
        if name in data:
            data[f"{name}_json"] = json.dumps(data.pop(name), default=str)
    return data


def _decode(data: dict, fields: tuple[str, ...]) -> dict:
    for name in fields:
        raw = data.pop(f"{name}_json", None)
        if raw is not None:
            data[name] = json.loads(raw)
    return data


def _encode_message(message: ChatMessage) -> dict:
    return _encode(message.model_dump(mode="python"), _JSON_FIELDS["message"])


def _decode_message(data: dict) -> ChatMessage:
    return ChatMessage.model_validate(_decode(data, _JSON_FIELDS["message"]))


def _encode_version(version: ResumeVersion) -> dict:
    return _encode(version.model_dump(mode="python"), _JSON_FIELDS["version"])


def _decode_version(data: dict) -> ResumeVersion:
    return ResumeVersion.model_validate(_decode(data, _JSON_FIELDS["version"]))


class FirestoreRepository(Repository):
    def __init__(self, client: Client, root_collection: str = "conversations") -> None:
        self._db = client
        self._root = root_collection

    # ------------------------------------------------------------------ helpers
    def _conv_ref(self, conversation_id: str):
        return self._db.collection(self._root).document(conversation_id)

    def _require(self, conversation_id: str):
        ref = self._conv_ref(conversation_id)
        snap = ref.get()
        if not snap.exists:
            raise NotFoundError(conversation_id)
        return ref, snap

    # ------------------------------------------------------------ conversations
    def save_conversation(self, conversation: Conversation) -> Conversation:
        self._conv_ref(conversation.id).set(conversation.model_dump(mode="python"))
        return conversation

    def get_conversation(self, conversation_id: str) -> Conversation:
        _, snap = self._require(conversation_id)
        return Conversation.model_validate(snap.to_dict())

    def list_conversations(self, limit: int = 50) -> list[Conversation]:
        query = (
            self._db.collection(self._root)
            .order_by("updated_at", direction=firestore.Query.DESCENDING)
            # The listing shows id/title/date/file only; skip memory slots such as a pasted JD.
            .select(["id", "title", "updated_at", "filename", "created_at"])
            .limit(limit)
        )
        return [Conversation.model_validate(s.to_dict()) for s in query.stream()]

    def delete_conversation(self, conversation_id: str) -> None:
        ref = self._conv_ref(conversation_id)
        for sub in ("messages", "versions"):
            for doc in ref.collection(sub).stream():
                doc.reference.delete()
        ref.delete()

    # ----------------------------------------------------------------- messages
    def add_message(self, conversation_id: str, message: ChatMessage) -> ChatMessage:
        ref, _ = self._require(conversation_id)
        ref.collection("messages").document(message.id).set(_encode_message(message))
        return message

    def list_messages(self, conversation_id: str) -> list[ChatMessage]:
        ref, _ = self._require(conversation_id)
        docs = ref.collection("messages").order_by("seq").stream()
        return [_decode_message(d.to_dict()) for d in docs]

    def recent_messages(self, conversation_id: str, limit: int) -> list[ChatMessage]:
        if not limit:
            return []
        query = (
            self._conv_ref(conversation_id)
            .collection("messages")
            .order_by("seq", direction=firestore.Query.DESCENDING)
            # History needs only what was said, not each reply's (large) agent trace.
            .select(["id", "seq", "role", "content", "created_at", "version_id", "client_id"])
            .limit(limit)
        )
        return [_decode_message(d.to_dict()) for d in query.stream()][::-1]

    def find_message(
        self, conversation_id: str, *, client_id: str | None = None, seq: int | None = None
    ) -> ChatMessage | None:
        query = self._conv_ref(conversation_id).collection("messages")
        if client_id is not None:
            query = query.where(filter=FieldFilter("client_id", "==", client_id))
        if seq is not None:
            query = query.where(filter=FieldFilter("seq", "==", seq))
        doc = next(iter(query.limit(1).stream()), None)
        return _decode_message(doc.to_dict()) if doc else None

    # ----------------------------------------------------------------- versions
    def add_version(self, conversation_id: str, version: ResumeVersion) -> ResumeVersion:
        ref, _ = self._require(conversation_id)
        ref.collection("versions").document(version.id).set(_encode_version(version))
        return version

    def get_version(self, conversation_id: str, version_id: str) -> ResumeVersion:
        ref, _ = self._require(conversation_id)
        snap = ref.collection("versions").document(version_id).get()
        if not snap.exists:
            raise NotFoundError(version_id)
        return _decode_version(snap.to_dict())

    def list_versions(self, conversation_id: str) -> list[ResumeVersion]:
        ref, _ = self._require(conversation_id)
        docs = ref.collection("versions").order_by("number").stream()
        return [_decode_version(d.to_dict()) for d in docs]

    def latest_version_number(self, conversation_id: str) -> int:
        query = (
            self._conv_ref(conversation_id)
            .collection("versions")
            .order_by("number", direction=firestore.Query.DESCENDING)
            .select(["number"])  # fetch one small field, not every resume body
            .limit(1)
        )
        doc = next(iter(query.stream()), None)
        return int(doc.to_dict()["number"]) if doc else 0


def create_repository(settings: Settings) -> Repository:
    if settings.storage_backend == "memory":
        from app.storage.memory import MemoryRepository

        logger.info("Using in-memory storage (data is lost on restart)")
        return MemoryRepository()
    repo = FirestoreRepository(build_client(settings))
    if settings.storage_cache:
        from app.storage.cache import CachedRepository

        logger.info("Write-through cache enabled in front of Firestore")
        return CachedRepository(repo)
    return repo

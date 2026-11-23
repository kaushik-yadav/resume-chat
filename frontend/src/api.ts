import type {
  ConversationListItem,
  ConversationState,
  ResumeVersion,
  SectionChange,
  StreamEvent,
  VersionSummary,
} from "./types";

const BASE = "/api";

export class ApiError extends Error {}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new ApiError(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

export const api = {
  createConversation: () =>
    request<{ id: string }>("/conversations", { method: "POST" }),
  listConversations: () => request<ConversationListItem[]>("/conversations"),
  getConversation: (id: string) => request<ConversationState>(`/conversations/${id}`),
  uploadResume: (id: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<ConversationState & { message: any }>(`/conversations/${id}/resume`, {
      method: "POST",
      body: form,
    });
  },
  listVersions: (id: string) => request<VersionSummary[]>(`/conversations/${id}/versions`),
  getVersion: (id: string, vid: string) =>
    request<ResumeVersion>(`/conversations/${id}/versions/${vid}`),
  undo: (id: string) => request<ConversationState>(`/conversations/${id}/undo`, { method: "POST" }),
  redo: (id: string) => request<ConversationState>(`/conversations/${id}/redo`, { method: "POST" }),
  restore: (id: string, vid: string) =>
    request<ConversationState>(`/conversations/${id}/versions/${vid}/restore`, { method: "POST" }),
  compare: (id: string, from: string, to: string) =>
    request<SectionChange[]>(
      `/conversations/${id}/compare?from=${encodeURIComponent(from)}&to=${encodeURIComponent(to)}`,
    ),
  exportUrl: (id: string, format: "docx" | "md", versionId?: string) =>
    `${BASE}/conversations/${id}/export?format=${format}${versionId ? `&version_id=${versionId}` : ""}`,
};

/**
 * POST a chat message and consume the Server-Sent Events stream.
 * (EventSource only supports GET, so the stream is parsed from fetch's body.)
 */
export async function streamChat(
  conversationId: string,
  message: string,
  jobDescription: string | null,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${BASE}/conversations/${conversationId}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ message, job_description: jobDescription || null }),
    signal,
  });
  if (!res.ok || !res.body) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new ApiError(String(detail));
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const data = block
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (data) onEvent(JSON.parse(data) as StreamEvent);
    }
  }
}

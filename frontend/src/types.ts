// Mirrors the backend's Pydantic models (app/models/*.py) and SSE event payloads.

export type SectionKind =
  | "header" | "summary" | "experience" | "education" | "skills" | "projects"
  | "certifications" | "awards" | "publications" | "languages" | "volunteer"
  | "interests" | "other";

export interface ResumeSection {
  id: string;
  title: string;
  kind: SectionKind;
  content: string;
}

export interface Resume {
  sections: ResumeSection[];
  language: string;
  locale: string | null;
}

export interface SectionChange {
  section_id: string;
  title: string;
  change_type: "modified" | "added" | "removed";
  before: string;
  after: string;
  words_added: number;
  words_removed: number;
}

export interface ResumeVersion {
  id: string;
  number: number;
  parent_id: string | null;
  created_at: string;
  source: "upload" | "agents" | "restore";
  label: string;
  agents: string[];
  resume: Resume;
  changes: SectionChange[];
}

export interface VersionSummary {
  id: string;
  number: number;
  parent_id: string | null;
  created_at: string;
  source: string;
  label: string;
  agents: string[];
  changed_sections: string[];
}

export interface ConversationContext {
  target_company: string | null;
  target_role: string | null;
  target_locale: string | null;
  job_description: string | null;
  last_action: string | null;
  summary: string;
}

export interface Conversation {
  id: string;
  title: string;
  filename: string | null;
  current_version_id: string | null;
  redo_stack: string[];
  context: ConversationContext;
  updated_at: string;
}

export interface ConversationListItem {
  id: string;
  title: string;
  updated_at: string;
  filename: string | null;
}

export interface AppliedEdit {
  section_id: string;
  title: string;
  rationale: string;
}

export interface AgentResult {
  agent: string;
  title: string;
  reasoning: string;
  edits: AppliedEdit[];
  insights: Record<string, any>;
  warnings: string[];
}

export interface PlanStep {
  agent: string;
  title?: string;
  instruction: string;
  target_sections: string[];
}

export interface MessageTrace {
  intent: string;
  reasoning: string;
  plan: PlanStep[];
  results: AgentResult[];
  llm: { calls: number; ms: number; input_tokens: number; output_tokens: number };
  ms: number;
}

export interface ChatMessage {
  id: string;
  seq: number;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  version_id: string | null;
  trace: MessageTrace | null;
}

export interface ConversationState {
  conversation: Conversation;
  current_version: ResumeVersion | null;
  can_undo: boolean;
  can_redo: boolean;
  messages?: ChatMessage[];
}

// ----- streaming events --------------------------------------------------------
export type StreamEvent =
  | { type: "user_message"; message: ChatMessage }
  | { type: "status"; stage: string; message: string }
  | { type: "plan"; intent: string; reasoning: string; steps: PlanStep[] }
  | { type: "agent_start"; agent: string; title: string; index: number; total: number; instruction: string }
  | { type: "progress"; agent: string; message: string }
  | ({ type: "agent_result" } & AgentResult)
  | { type: "token"; text: string }
  | { type: "version"; version: ResumeVersion; can_undo: boolean; can_redo: boolean }
  | ({ type: "done"; message: ChatMessage } & ConversationState)
  | { type: "error"; message: string };

/** What the UI shows while a turn is in flight. */
export interface LiveTurn {
  status: string;
  plan: { intent: string; reasoning: string; steps: PlanStep[] } | null;
  activeAgent: string | null;
  progress: { agent: string; message: string }[];
  results: AgentResult[];
  reply: string;
}

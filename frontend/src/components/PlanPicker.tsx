import { useEffect, useRef, useState } from "react";
import { Check, ChevronDown, Gauge } from "lucide-react";
import type { Plan } from "../types";

const ROLES = [
  { id: "router", label: "Routing", hint: "Understands the request" },
  { id: "agent", label: "Agents", hint: "Rewrites the resume" },
  { id: "reply", label: "Reply", hint: "Writes the chat answer" },
] as const;

function usd(v: number | null) {
  return v == null ? "n/a" : `$${v.toFixed(2)}`;
}

/** Plan switcher: which models serve this conversation, as comparable cards. */
export default function PlanPicker({
  plans,
  current,
  disabled,
  onChange,
}: {
  plans: Plan[];
  current: string;
  disabled: boolean;
  onChange: (plan: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  const active = plans.find((p) => p.id === current);

  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => {
      if (!box.current?.contains(e.target as Node)) setOpen(false);
    };
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", esc);
    };
  }, [open]);

  if (!plans.length) return null;
  return (
    <div className="plan-picker" ref={box}>
      <button
        className={`btn small plan-btn plan-${current}`}
        disabled={disabled}
        aria-expanded={open}
        aria-haspopup="dialog"
        onClick={() => setOpen((v) => !v)}
        title="Choose the plan (models) for this conversation"
      >
        <Gauge size={14} /> {active?.name ?? current}
        <ChevronDown size={14} className={open ? "is-open" : undefined} />
      </button>
      {open && (
        <div className="plan-menu" role="dialog" aria-label="Plans">
          <header className="plan-menu-head">
            <h3>Plans</h3>
            <p>Each plan decides which model runs each step. Prices are per 1M tokens.</p>
          </header>
          <div className="plan-cards">
            {plans.map((p) => {
              const on = p.id === current;
              return (
                <button
                  key={p.id}
                  className={`plan-card${on ? " is-on" : ""}`}
                  aria-pressed={on}
                  onClick={() => {
                    setOpen(false);
                    if (!on) onChange(p.id);
                  }}
                >
                  <div className="pc-head">
                    <h4>{p.name}</h4>
                    {on && (
                      <span className="pc-current">
                        <Check size={12} aria-hidden /> Current
                      </span>
                    )}
                  </div>
                  <p className="pc-tagline">{p.tagline}</p>
                  <dl className="pc-roles">
                    {ROLES.map((role) => {
                      const price = p.prices[role.id];
                      return (
                        <div key={role.id} className="pc-role">
                          <dt title={role.hint}>{role.label}</dt>
                          <dd>
                            <span className="pc-model">{price.model}</span>
                            {role.id === "agent" && p.agent_effort !== "low" && (
                              <span className="pc-effort">{p.agent_effort} reasoning</span>
                            )}
                            <span className="pc-price">
                              <span>
                                <b>{usd(price.input_per_1m)}</b> in
                              </span>
                              <span>
                                <b>{usd(price.output_per_1m)}</b> out
                              </span>
                            </span>
                          </dd>
                        </div>
                      );
                    })}
                  </dl>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

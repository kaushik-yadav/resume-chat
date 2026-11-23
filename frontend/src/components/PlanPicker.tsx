import { useEffect, useRef, useState } from "react";
import { Check, ChevronDown, Gauge } from "lucide-react";
import type { Plan } from "../types";

function price(p: { input_per_1m: number | null; output_per_1m: number | null }) {
  return p.input_per_1m == null ? "n/a" : `$${p.input_per_1m} / $${p.output_per_1m}`;
}

/** Plan switcher: which models serve this conversation (free = lite model everywhere). */
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
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);

  if (!plans.length) return null;
  return (
    <div className="plan-picker" ref={box}>
      <button
        className={`btn small plan-btn plan-${current}`}
        disabled={disabled}
        onClick={() => setOpen((v) => !v)}
        title="Choose the plan (models) for this conversation"
      >
        <Gauge size={14} /> {active?.name ?? current} <ChevronDown size={13} />
      </button>
      {open && (
        <div className="plan-menu" role="menu">
          <div className="plan-menu-head">
            Plans decide which model runs each step. Prices are per 1M tokens (input / output).
          </div>
          {plans.map((p) => (
            <button
              key={p.id}
              role="menuitemradio"
              aria-checked={p.id === current}
              className={`plan-row ${p.id === current ? "on" : ""}`}
              onClick={() => {
                setOpen(false);
                if (p.id !== current) onChange(p.id);
              }}
            >
              <div className="plan-row-head">
                <b>{p.name}</b>
                {p.id === current && <Check size={14} />}
                <span className="sub">{p.tagline}</span>
              </div>
              <table>
                <tbody>
                  {(["router", "agent", "reply"] as const).map((role) => (
                    <tr key={role}>
                      <td className="sub">{role === "agent" ? "Agents" : role === "router" ? "Routing" : "Reply"}</td>
                      <td><code>{p.prices[role].model}</code>{role === "agent" && p.agent_effort !== "low" && ` · ${p.agent_effort} reasoning`}</td>
                      <td className="sub">{price(p.prices[role])}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

import { useEffect, useRef, useState } from "react";
import { Coins } from "lucide-react";

/**
 * Token counter that moves while a turn runs (an estimate) and settles on the real count.
 *
 * Providers report token usage only when a call finishes, so during a turn the value is an
 * estimate (completed calls' real usage + streamed characters / 4 + elapsed time on running
 * steps). When the turn ends the meter eases from the estimate to the actual number.
 */
export default function TokenMeter({
  value,
  live,
  cost,
  compact = false,
}: {
  value: number;
  live: boolean;
  cost?: number | null;
  compact?: boolean;
}) {
  const [shown, setShown] = useState(value);
  const from = useRef(value);
  const frame = useRef<number>(0);

  useEffect(() => {
    const start = performance.now();
    const origin = from.current;
    const duration = live ? 350 : 700;
    cancelAnimationFrame(frame.current);
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - (1 - t) ** 3;
      const next = Math.round(origin + (value - origin) * eased);
      setShown(next);
      from.current = next;
      if (t < 1) frame.current = requestAnimationFrame(tick);
    };
    frame.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame.current);
  }, [value, live]);

  return (
    <span
      className={`token-meter${live ? " is-live" : ""}${compact ? " is-compact" : ""}`}
      title={live ? "Estimated while the reply is generated; exact count shown when it finishes" : "Exact tokens used"}
    >
      <Coins size={13} aria-hidden />
      <span className="tm-value">
        {live ? "≈ " : ""}
        {shown.toLocaleString()}
      </span>
      <span className="tm-unit">tokens</span>
      {cost != null && cost > 0 && <span className="tm-cost">${cost < 0.01 ? cost.toFixed(4) : cost.toFixed(3)}</span>}
    </span>
  );
}

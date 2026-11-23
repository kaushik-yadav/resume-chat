import { useEffect, useState } from "react";

type Health = { status: string; llm_provider: string; storage_backend: string };

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  return (
    <main className="placeholder">
      <h1>Resume Chat</h1>
      <p>{health ? `API ${health.status} · ${health.llm_provider}` : "Connecting to API…"}</p>
    </main>
  );
}

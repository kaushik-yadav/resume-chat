import { useRef, useState } from "react";
import { FileUp, Loader2 } from "lucide-react";

interface Props {
  onUpload: (file: File) => Promise<void>;
}

const ACCEPT = ".pdf,.docx";

export default function UploadPane({ onUpload }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handle(file: File | undefined) {
    if (!file) return;
    if (!/\.(pdf|docx)$/i.test(file.name)) {
      setError("Please upload a PDF or DOCX file.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await onUpload(file);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="upload">
      <div
        className={`dropzone ${over ? "over" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          handle(e.dataTransfer.files[0]);
        }}
      >
        <FileUp className="icon" size={40} />
        <h2>Chat with your resume</h2>
        <p>Upload a resume and ask for what you need. Specialist agents research companies,
          match job descriptions, translate for new markets and sharpen individual sections.</p>
        {error && <div className="error-banner">{error}</div>}
        <button className="btn primary" disabled={busy} onClick={() => input.current?.click()}>
          {busy ? <Loader2 size={16} className="spin" /> : <FileUp size={16} />}
          {busy ? "Parsing resume…" : "Upload PDF or DOCX"}
        </button>
        <input
          ref={input}
          type="file"
          accept={ACCEPT}
          hidden
          onChange={(e) => handle(e.target.files?.[0])}
        />
        <div className="hint">or drag and drop it here · max 10 MB</div>
        <div className="features">
          <div><b>Company research</b>"Optimize my resume for Google"</div>
          <div><b>Job matching</b>"Match this job description" + score &amp; gaps</div>
          <div><b>Localization</b>"Translate to Spanish for Mexico"</div>
          <div><b>Section polish</b>"Make my skills fit a data scientist role"</div>
        </div>
      </div>
    </div>
  );
}

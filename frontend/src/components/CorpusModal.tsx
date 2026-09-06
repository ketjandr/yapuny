// Corpus manager: a header button (in the training pane) that opens a modal to upload, preview, and
// edit the single training corpus. The corpus is one plain .txt on the worker's disk. Edits (typed
// or loaded from an uploaded file) stay in the editor until Commit writes them to disk; closing with
// uncommitted edits prompts a confirmation so nothing is lost silently.
import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useTooltip } from "@/components/tooltipContext";
import { api } from "@/lib/api";
import { toast } from "@/store/toastStore";

type CommitState = "idle" | "committing" | "committed" | "error";

const fmtBytes = (n: number) => (n < 1024 ? `${n} B` : n < 1024 * 1024 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1024 / 1024).toFixed(2)} MB`);

export function CorpusButton() {
  const [open, setOpen] = useState(false);
  const tip = useTooltip("Upload / edit corpus");
  return (
    <>
      <button type="button" className="pane-icon" aria-label="Upload / edit corpus" onClick={() => setOpen(true)} {...tip}>
        ≣
      </button>
      {open && <CorpusModal onClose={() => setOpen(false)} />}
    </>
  );
}

function CorpusModal({ onClose }: { onClose: () => void }) {
  const [text, setText] = useState("");
  const [committed, setCommitted] = useState(""); // last on-disk content (dirty = text !== committed)
  const [exists, setExists] = useState(false);
  const [loading, setLoading] = useState(true);
  const [commit, setCommit] = useState<CommitState>("idle");
  const [confirmClose, setConfirmClose] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const dirty = text !== committed;

  // live stats from the editor content (no round-trip needed for feedback)
  const bytes = new TextEncoder().encode(text).length;
  const lines = text ? text.split("\n").length : 0;

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const d = await api.getCorpus();
      setText(d.text ?? "");
      setCommitted(d.text ?? "");
      setExists(true);
    } catch {
      setText(""); // 404 = no corpus yet
      setCommitted("");
      setExists(false);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const onEdit = (v: string) => {
    setText(v);
    if (commit !== "idle") setCommit("idle");
  };

  // upload reads the file into the editor (uncommitted); Commit is what writes it to disk
  const onUpload = async (file: File) => {
    if (!file.name.endsWith(".txt")) {
      toast.error("Only .txt files are accepted");
      return;
    }
    try {
      const content = await file.text();
      setText(content);
      setCommit("idle");
    } catch (e) {
      toast.error(`Couldn't read file: ${(e as Error).message}`);
    }
  };

  const onCommit = async () => {
    setCommit("committing");
    try {
      await api.saveCorpus(text);
      setCommitted(text);
      setExists(true);
      setCommit("committed");
    } catch (e) {
      setCommit("error");
      toast.error(`Couldn't commit corpus: ${(e as Error).message}`);
    }
  };

  // clear all is just an edit (empties the editor); it only reaches disk once committed
  const onClearAll = () => {
    setText("");
    setCommit("idle");
  };

  const requestClose = () => {
    if (dirty) setConfirmClose(true);
    else onClose();
  };

  const commitLabel =
    commit === "committing" ? "committing…" : commit === "error" ? "commit failed" : dirty ? "uncommitted changes" : exists ? "committed" : "";

  return createPortal(
    <div className="corpus-overlay" onClick={(e) => { if (e.target === e.currentTarget) requestClose(); }}>
      <div className="corpus-modal" role="dialog" aria-label="Training corpus">
        <div className="corpus-head">
          <span className="corpus-title">Training corpus</span>
          <div className="corpus-head-tools">
            <input
              ref={fileRef}
              type="file"
              accept=".txt"
              hidden
              onChange={(e) => { const f = e.target.files?.[0]; if (f) onUpload(f); e.target.value = ""; }}
            />
            <button type="button" className="pane-icon" aria-label="Upload .txt" title="Upload .txt" onClick={() => fileRef.current?.click()}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </button>
            <button type="button" className="btn tiny primary" disabled={!dirty || commit === "committing"} onClick={onCommit}>
              {commit === "committing" ? "committing…" : "Commit"}
            </button>
            <button type="button" className="btn tiny danger" disabled={text.length === 0} onClick={onClearAll}>Clear all</button>
            <button type="button" className="pane-icon close" aria-label="Close" onClick={requestClose}>✕</button>
          </div>
        </div>

        <div className="corpus-meta">
          {loading ? (
            <span>loading…</span>
          ) : (
            <>
              <span>{fmtBytes(bytes)}</span>
              <span>{text.length.toLocaleString()} chars</span>
              <span>{lines.toLocaleString()} lines</span>
              {commitLabel && <span className={`corpus-save ${dirty ? "dirty" : commit}`}>{commitLabel}</span>}
            </>
          )}
        </div>

        <textarea
          className="corpus-text mono"
          spellCheck={false}
          placeholder={loading ? "" : "Paste or type your corpus here, or upload a .txt file. Press Commit to save to disk."}
          value={text}
          onChange={(e) => onEdit(e.target.value)}
          disabled={loading}
        />
        <div className="corpus-note">Edits are saved to the worker only when you commit. One corpus at a time; every model trains its own tokenizer on it.</div>

        {confirmClose && (
          <div className="corpus-confirm" onClick={(e) => { if (e.target === e.currentTarget) setConfirmClose(false); }}>
            <div className="corpus-confirm-box" role="alertdialog" aria-label="Discard changes?">
              <div className="corpus-confirm-title">Discard uncommitted changes?</div>
              <div className="corpus-confirm-msg">Your edits haven't been committed to disk. Close anyway?</div>
              <div className="corpus-confirm-actions">
                <button type="button" className="btn tiny" onClick={() => setConfirmClose(false)}>Keep editing</button>
                <button type="button" className="btn tiny danger" onClick={onClose}>Discard</button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>,
    document.body,
  );
}

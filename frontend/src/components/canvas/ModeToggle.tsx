import { type CanvasMode, useCanvasStore } from "@/store/canvasStore";

const MODES: { id: CanvasMode; label: string }[] = [
  { id: "train", label: "Train" },
  { id: "inference", label: "Inference" },
];

export function ModeToggle() {
  const mode = useCanvasStore((s) => s.mode);
  const setMode = useCanvasStore((s) => s.setMode);

  return (
    <div className="mode-toggle" role="tablist" aria-label="Canvas mode">
      {MODES.map((m) => (
        <button
          key={m.id}
          type="button"
          role="tab"
          aria-selected={mode === m.id}
          className={mode === m.id ? "a" : ""}
          onClick={() => setMode(m.id)}
        >
          {m.label}
        </button>
      ))}
    </div>
  );
}

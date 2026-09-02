// Toast stack, pinned top-center over everything. Fed by toastStore.
import { useToastStore } from "@/store/toastStore";

export function ToastContainer() {
  const toasts = useToastStore((s) => s.toasts);
  const dismiss = useToastStore((s) => s.dismiss);

  return (
    <div className="toast-stack" role="region" aria-live="polite">
      {toasts.map((t) => (
        <div
          key={t.id}
          className={`toast toast-${t.kind}${t.leaving ? " toast-leaving" : ""}`}
          role="alert"
        >
          <span className="toast-msg">{t.message}</span>
          <button
            className="toast-x"
            type="button"
            onClick={() => dismiss(t.id)}
            aria-label="Dismiss"
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );
}

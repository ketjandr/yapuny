// The canvas keyboard shortcut table. Add a new shortcut here (one row) - dispatch is
// handled by useShortcuts. Delete/Backspace is left to React Flow's built-in handling.
import { useMemo } from "react";
import { type Shortcut, useShortcuts } from "@/lib/shortcuts";
import { useCanvasStore } from "@/store/canvasStore";

export function useCanvasShortcuts(): void {
  const shortcuts = useMemo<Shortcut[]>(
    () => [
      {
        id: "copy",
        keys: "mod+c",
        run: () => {
          const s = useCanvasStore.getState();
          const nodeIds = s.nodes.filter((n) => n.selected).map((n) => n.id);
          const edgeIds = s.edges.filter((e) => e.selected).map((e) => e.id);
          if (nodeIds.length) s.copyNodes(nodeIds, edgeIds);
        },
      },
      {
        id: "paste",
        keys: "mod+v",
        run: () => useCanvasStore.getState().pasteAtOffset(),
      },
    ],
    [],
  );

  useShortcuts(shortcuts);
}

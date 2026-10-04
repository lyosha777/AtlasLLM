import { useCallback, useEffect, useRef, useState } from "react";
import { getModels, type ModelCatalog } from "./api";


export function useModels() {
  const [catalog, setCatalog] = useState<ModelCatalog | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const controller = useRef<AbortController | null>(null);
  const load = useCallback(() => {
    controller.current?.abort();
    const request = new AbortController();
    controller.current = request;
    void getModels(request.signal).then((result) => {
      if (request.signal.aborted) return;
      setCatalog(result);
      setError(null);
    }).catch((error: unknown) => {
      if (request.signal.aborted) return;
      setCatalog(null);
      setError(error instanceof Error ? error.message : "Couldn't discover local models.");
    }).finally(() => {
      if (!request.signal.aborted) setLoading(false);
    });
  }, []);
  const refresh = useCallback(() => { setLoading(true); load(); }, [load]);
  useEffect(() => {
    load();
    const focus = () => refresh();
    window.addEventListener("focus", focus);
    return () => { controller.current?.abort(); window.removeEventListener("focus", focus); };
  }, [load, refresh]);
  return { catalog, loading, error, refresh };
}

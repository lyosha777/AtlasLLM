import { useEffect, useState, useSyncExternalStore } from "react";

import { GenerationStore } from "./generationStore";


export function useGenerations() {
  const [store] = useState(() => new GenerationStore());
  const state = useSyncExternalStore(store.subscribe, store.getSnapshot);
  useEffect(() => {
    const controller = new AbortController();
    void store.connect(controller.signal);
    return () => controller.abort();
  }, [store]);
  return { store, ...state };
}

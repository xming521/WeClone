import { create } from 'zustand';

/** A short notice of what an action did, with an undo when it can be taken back. */
export interface Toast {
  id: number;
  message: string;
  tone?: 'error';
  undo?: () => void;
}

interface ToastState {
  current: Toast | null;
  show: (toast: Omit<Toast, 'id'>) => number;
  dismiss: (id?: number) => void;
  /** Takes back the action of the notice on screen; false when it has nothing to take back. */
  undo: () => boolean;
}

let count = 0;

// One notice at a time: a newer action replaces the older notice, and with it the older undo.
export const useToasts = create<ToastState>((set, get) => ({
  current: null,
  show: (toast) => {
    const id = ++count;
    set({ current: { ...toast, id } });
    return id;
  },
  dismiss: (id) => {
    if (id === undefined || get().current?.id === id) set({ current: null });
  },
  undo: () => {
    const undo = get().current?.undo;
    if (!undo) return false;
    set({ current: null });
    undo();
    return true;
  },
}));

export const toast = (message: string, options: Omit<Toast, 'id' | 'message'> = {}) =>
  useToasts.getState().show({ message, ...options });

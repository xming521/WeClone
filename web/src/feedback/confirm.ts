import { create } from 'zustand';

/** A question put to the owner before an action, asked in the page's own dialog instead of the browser's. */
export interface Question {
  /** The question itself, short: what is about to happen. */
  title: string;
  /** What follows from it, when the title alone doesn't say. */
  body?: string;
  /** The name of the action, on the button that goes ahead with it. */
  action: string;
  /** The other button, when a plain Cancel would read as the action itself. */
  dismiss?: string;
  tone?: 'danger' | 'adopt';
}

export interface Confirmation extends Question {
  id: number;
  answer: (ok: boolean) => void;
}

let count = 0;

export const useConfirmations = create<{ current: Confirmation | null }>(() => ({ current: null }));

/** Resolves true when the owner goes ahead, false when they cancel. One question at a time: a newer one answers the
 * older with false. */
export function confirm(question: Question): Promise<boolean> {
  const previous = useConfirmations.getState().current;
  const id = ++count;
  return new Promise((resolve) => {
    const answer = (ok: boolean) => {
      if (useConfirmations.getState().current?.id === id) useConfirmations.setState({ current: null });
      resolve(ok);
    };
    useConfirmations.setState({ current: { ...question, id, answer } });
    previous?.answer(false);
  });
}

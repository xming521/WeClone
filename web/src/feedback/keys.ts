// Inputs a key press can't type into: shortcuts still apply while one of them has focus.
const PRESSABLE = new Set(['checkbox', 'radio', 'button', 'submit', 'reset']);

/** Whether a key press belongs to a text field or an open dialog rather than to the page's shortcuts. */
export function isTyping(event: KeyboardEvent): boolean {
  if (document.querySelector('dialog[open]')) return true;
  const target = event.target as HTMLElement | null;
  if (target instanceof HTMLInputElement) return !PRESSABLE.has(target.type);
  return !!target?.closest('textarea, select, [contenteditable="true"], [role="combobox"], [role="listbox"]');
}

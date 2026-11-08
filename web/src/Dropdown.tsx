import * as Popover from '@radix-ui/react-popover';
import * as Select from '@radix-ui/react-select';
import { ChevronDown } from 'lucide-react';
import { useCallback, useEffect, useId, useRef, useState, type InputHTMLAttributes } from 'react';
import { useTranslation } from 'react-i18next';
import './dropdown.css';

/** Where a field's menu goes: inside a modal dialog it has to be in the dialog, which makes the rest of the page
 * inert. Elsewhere null, which is the page's body. */
function useMenuContainer() {
  const [container, setContainer] = useState<HTMLElement | null>(null);
  const ref = useCallback((node: HTMLElement | null) => setContainer(node?.closest('dialog') ?? null), []);
  return [container, ref] as const;
}

/** A list to pick from in a text field's box, showing a placeholder until something is picked. */
export function SelectInput({
  id,
  value,
  onChange,
  options,
  placeholder,
  disabled,
  required,
  invalid,
}: {
  id?: string;
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string }[];
  placeholder?: string;
  disabled?: boolean;
  required?: boolean;
  invalid?: boolean;
}) {
  const [container, ref] = useMenuContainer();
  return (
    <Select.Root value={value} onValueChange={onChange} disabled={disabled} required={required}>
      <Select.Trigger id={id} ref={ref} className="wc-input dropdown-select" aria-invalid={invalid || undefined}>
        <Select.Value placeholder={placeholder} />
        <ChevronDown className="dropdown-chevron" size={14} strokeWidth={1.8} aria-hidden="true" />
      </Select.Trigger>
      <Select.Portal container={container}>
        <Select.Content className="wc-menu dropdown-menu" position="popper" side="bottom" align="start" sideOffset={6}>
          <Select.Viewport>
            {options.map((option) => (
              <Select.Item
                key={option.value}
                value={option.value}
                className="wc-menu__item"
                aria-checked={option.value === value}
              >
                <Select.ItemText>{option.label}</Select.ItemText>
              </Select.Item>
            ))}
          </Select.Viewport>
        </Select.Content>
      </Select.Portal>
    </Select.Root>
  );
}

/** A text field that also offers values: the list opens on a click, the chevron or ↓, and narrows to what is typed.
 * Any other value can still be typed. */
export function ComboInput({
  value,
  onChange,
  options,
  invalid,
  ...input
}: {
  value: string;
  onChange: (value: string) => void;
  options: string[];
  invalid?: boolean;
} & Omit<InputHTMLAttributes<HTMLInputElement>, 'value' | 'onChange'>) {
  const { t } = useTranslation();
  const listId = useId();
  const [container, ref] = useMenuContainer();
  const [open, setOpen] = useState(false);
  // What was typed since the list opened; until then it shows every option.
  const [query, setQuery] = useState<string | null>(null);
  const [active, setActive] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const needle = query?.trim().toLocaleLowerCase();
  const shown = needle ? options.filter((option) => option.toLocaleLowerCase().includes(needle)) : options;
  const expanded = open && shown.length > 0;

  useEffect(() => {
    if (expanded && active >= 0) listRef.current?.children[active]?.scrollIntoView({ block: 'nearest' });
  }, [expanded, active]);

  function show() {
    setOpen(true);
    setQuery(null);
    setActive(options.indexOf(value));
  }
  function close() {
    setOpen(false);
    setQuery(null);
  }
  function pick(option: string) {
    onChange(option);
    close();
  }

  return (
    <Popover.Root open={expanded} onOpenChange={(next) => !next && close()}>
      <Popover.Anchor asChild>
        <div
          ref={ref}
          className="wc-input dropdown-combo"
          data-state={expanded ? 'open' : 'closed'}
          aria-invalid={invalid || undefined}
        >
          <input
            {...input}
            ref={inputRef}
            value={value}
            role="combobox"
            aria-expanded={expanded}
            aria-controls={expanded ? listId : undefined}
            aria-autocomplete="list"
            aria-activedescendant={expanded && active >= 0 ? `${listId}-${active}` : undefined}
            onChange={(event) => {
              onChange(event.target.value);
              setOpen(true);
              setQuery(event.target.value);
              setActive(-1);
            }}
            onClick={() => {
              if (!open) show();
            }}
            onBlur={close}
            onKeyDown={(event) => {
              if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
                event.preventDefault();
                const step = event.key === 'ArrowDown' ? 1 : -1;
                if (!expanded) show();
                else if (active < 0) setActive(step > 0 ? 0 : shown.length - 1);
                else setActive((active + step + shown.length) % shown.length);
              } else if (event.key === 'Enter' && expanded && shown[active] !== undefined) {
                event.preventDefault();
                pick(shown[active]);
              }
            }}
          />
          <button
            type="button"
            className="dropdown-toggle"
            tabIndex={-1}
            disabled={input.disabled}
            aria-label={t(($) => $.common.showOptions)}
            // Keeps the caret in the field.
            onMouseDown={(event) => event.preventDefault()}
            onClick={() => (expanded ? close() : show())}
          >
            <ChevronDown className="dropdown-chevron" size={14} strokeWidth={1.8} aria-hidden="true" />
          </button>
        </div>
      </Popover.Anchor>
      <Popover.Portal container={container}>
        <Popover.Content
          ref={listRef}
          id={listId}
          role="listbox"
          className="wc-menu dropdown-menu"
          side="bottom"
          align="start"
          sideOffset={6}
          onOpenAutoFocus={(event) => event.preventDefault()}
          onCloseAutoFocus={(event) => event.preventDefault()}
          // The field and its chevron open and close the list themselves.
          onPointerDownOutside={(event) => {
            if (inputRef.current?.parentElement?.contains(event.target as Node)) event.preventDefault();
          }}
          // A press on the list, its scrollbar included, leaves the focus in the field.
          onMouseDown={(event) => event.preventDefault()}
        >
          {shown.map((option, index) => (
            <div
              key={option}
              id={`${listId}-${index}`}
              role="option"
              className="wc-menu__item"
              aria-selected={index === active}
              aria-checked={option === value}
              data-highlighted={index === active ? '' : undefined}
              title={option}
              onMouseMove={() => setActive(index)}
              onClick={() => pick(option)}
            >
              <span>{option}</span>
            </div>
          ))}
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

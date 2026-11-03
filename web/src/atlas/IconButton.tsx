import * as Tooltip from '@radix-ui/react-tooltip';

/**
 * DS Tooltip around one trigger: no arrow, two shrinking tail dots (`data-tail`) point at it. The tail
 * follows the side the bubble opens on, so collisions must not flip it. `top` serves the toolbar, with
 * the tail at the start or the end edge; `right` serves the review rail.
 */
export function Tip({
  content,
  side = 'top',
  align = 'start',
  children,
}: {
  content: React.ReactNode;
  side?: 'top' | 'right';
  align?: 'start' | 'center' | 'end';
  children: React.ReactElement;
}) {
  const tail = side === 'right' ? 'left' : align === 'end' ? 'bottom-right' : 'bottom-left';
  return (
    <Tooltip.Root delayDuration={400}>
      <Tooltip.Trigger asChild>{children}</Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content
          className="wc-tip"
          data-tail={tail}
          side={side}
          align={align}
          sideOffset={18}
          avoidCollisions={false}
        >
          {content}
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}

export function IconButton({
  label,
  children,
  onClick,
  align,
}: {
  label: string;
  children: React.ReactNode;
  onClick: () => void;
  /** Which edge of the button the bubble lines up with; `end` for the last button of a bar. */
  align?: 'start' | 'end';
}) {
  return (
    <Tip content={label} align={align}>
      <button className="wc-iconbtn" type="button" aria-label={label} onClick={onClick}>
        {children}
      </button>
    </Tip>
  );
}

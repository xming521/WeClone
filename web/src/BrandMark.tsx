import mark from './assets/logo/weclone-mark.svg';
import markSmall from './assets/logo/weclone-mark-small.svg';
import favicon from './assets/logo/weclone-favicon.svg';

/** The WeClone mark from the design system: `mark` at 40px and up, `small` at 24–32px, `favicon` (square) at 16–28px. */
const VARIANTS = { mark, small: markSmall, favicon };

export function BrandMark({ className, variant = 'mark' }: { className?: string; variant?: keyof typeof VARIANTS }) {
  return (
    <img
      className={className ? `brand-mark ${className}` : 'brand-mark'}
      src={VARIANTS[variant]}
      alt=""
      draggable={false}
    />
  );
}

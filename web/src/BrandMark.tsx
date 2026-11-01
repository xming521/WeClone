import logo from './assets/logo.png';

export function BrandMark({ className }: { className?: string }) {
  return <img className={className ? `brand-mark ${className}` : 'brand-mark'} src={logo} alt="" draggable={false} />;
}

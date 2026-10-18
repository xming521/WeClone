import type { CSSProperties } from 'react';
import type { LucideIcon } from 'lucide-react';
import { ChevronLeft, Compass, Fingerprint, Heart, Layers3, Target } from 'lucide-react';
import { DIMENSIONS, chineseNumeral, dimensionColor, type DimensionId } from '../data';
import { useProfileStore } from '../store';

const DIMENSION_ICONS: Record<DimensionId, LucideIcon> = {
  1: Fingerprint,
  3: Target,
  4: Layers3,
  5: Compass,
  8: Heart,
};

export function LeftPanel({
  open,
  onClose,
  onInteract,
  onSelect,
}: {
  open: boolean;
  onClose: () => void;
  onInteract: () => void;
  onSelect: () => void;
}) {
  const viewModel = useProfileStore((state) => state.viewModel)!;
  const focusedDimension = useProfileStore((state) => state.focusedDimension);
  const reveal = useProfileStore((state) => state.reveal);
  return (
    <aside
      className={`left-panel${open ? '' : ' is-collapsed'}`}
      aria-hidden={!open}
      onPointerDownCapture={onInteract}
      onFocusCapture={onInteract}
    >
      <div className="brand-row">
        <span className="brand-symbol">✳</span>
        <span>
          WeClone<span className="brand-separator">/</span>Profile Atlas
        </span>
        <button className="sidebar-close" type="button" aria-label="收起左栏" onClick={onClose}>
          <ChevronLeft size={17} />
        </button>
      </div>
      <div className="intro-block">
        <div className="eyebrow">
          <span className="eyebrow-line" /> A MAP OF YOU
        </div>
        <h1>
          每一条线索，
          <br />
          <em>都更了解你。</em>
        </h1>
        <p>沿着{chineseNumeral(DIMENSIONS.length)}条生活脉络，探索你的画像。</p>
      </div>
      <div className="nav-heading">画像维度</div>
      <nav className="dimension-list" aria-label="画像维度">
        {viewModel.dimensionIds.map((id) => {
          const node = viewModel.nodes.get(id)!;
          const color = dimensionColor(node.dim);
          const Icon = DIMENSION_ICONS[node.dim as DimensionId];
          return (
            <button
              key={id}
              type="button"
              className={`dimension-item ${focusedDimension === node.dim ? 'active' : ''}`}
              onClick={() => {
                reveal(id);
                onSelect();
              }}
              style={{ '--dimension-color': color } as CSSProperties}
            >
              <span className="dimension-icon">
                <Icon size={19} strokeWidth={1.7} />
              </span>
              <span className="dimension-copy">
                <strong>{node.name}</strong>
                <small>{node.attributeCount} 项画像属性</small>
              </span>
            </button>
          );
        })}
      </nav>
      <div className="left-bottom">
        <div className="data-footnote">
          <span className="live-dot" /> {viewModel.facts.length} 条画像
        </div>
      </div>
    </aside>
  );
}

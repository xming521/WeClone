import type { CSSProperties } from 'react';
import type { LucideIcon } from 'lucide-react';
import { ChevronLeft, Compass, Fingerprint, Heart, Layers3, Target } from 'lucide-react';
import { DIMENSIONS, chineseNumeral, dimensionColor, displayProfileText, type DimensionId } from '../data';
import type { Grouping } from '../data';
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
  const fullModel = useProfileStore((state) => state.fullModel)!;
  const hasDomains = useProfileStore((state) => !!state.input?.domains?.length);
  const focusedDimension = useProfileStore((state) => state.focusedDimension);
  const reveal = useProfileStore((state) => state.reveal);
  const byDomain = viewModel.grouping === 'domain';
  const heading = byDomain ? '画像领域' : '画像维度';
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
        <p>
          {byDomain
            ? `沿着${chineseNumeral(fullModel.dimensionIds.length)}个生活领域，探索你的画像。`
            : `沿着${chineseNumeral(DIMENSIONS.length)}条生活脉络，探索你的画像。`}
        </p>
      </div>
      <div className="nav-heading-row">
        <div className="nav-heading">{heading}</div>
        {hasDomains && <GroupingSwitch value={viewModel.grouping} />}
      </div>
      <nav className={`dimension-list${byDomain ? ' is-compact' : ''}`} aria-label={heading}>
        {viewModel.dimensionIds.map((id) => {
          const node = viewModel.nodes.get(id)!;
          const color = dimensionColor(node.dim);
          const Icon = byDomain ? null : DIMENSION_ICONS[node.dim as DimensionId];
          return (
            <button
              key={id}
              type="button"
              className={`dimension-item ${focusedDimension === node.dim ? 'active' : ''}`}
              title={node.description ? displayProfileText(node.description) : undefined}
              onClick={() => {
                reveal(id);
                onSelect();
              }}
              style={{ '--dimension-color': color } as CSSProperties}
            >
              <span className="dimension-icon" aria-hidden="true">
                {Icon ? (
                  <Icon size={19} strokeWidth={1.7} />
                ) : (
                  <span className="domain-initial">{[...node.name][0]}</span>
                )}
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

const GROUPINGS: { value: Grouping; label: string }[] = [
  { value: 'dimension', label: '按维度' },
  { value: 'domain', label: '按领域' },
];

function GroupingSwitch({ value }: { value: Grouping }) {
  const setGrouping = useProfileStore((state) => state.setGrouping);
  return (
    <div className="grouping-switch" role="group" aria-label="画像分组方式">
      {GROUPINGS.map((option) => (
        <button
          key={option.value}
          type="button"
          aria-pressed={value === option.value}
          onClick={() => setGrouping(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

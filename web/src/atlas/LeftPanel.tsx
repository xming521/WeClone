import type { CSSProperties } from 'react';
import type { LucideIcon } from 'lucide-react';
import { ChevronLeft, Compass, Fingerprint, Heart, Layers3, Target } from 'lucide-react';
import { Trans, useTranslation } from 'react-i18next';
import { DIMENSIONS, dimensionColor, displayProfileText, type DimensionId } from '../data';
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
  const { t } = useTranslation();
  const viewModel = useProfileStore((state) => state.viewModel)!;
  const fullModel = useProfileStore((state) => state.fullModel)!;
  const focusedDimension = useProfileStore((state) => state.focusedDimension);
  const reveal = useProfileStore((state) => state.reveal);
  const byDomain = viewModel.grouping === 'domain';
  const heading = byDomain ? t(($) => $.atlas.left.domainsTitle) : t(($) => $.atlas.left.dimensionsTitle);
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
        <button className="sidebar-close" type="button" aria-label={t(($) => $.atlas.left.close)} onClick={onClose}>
          <ChevronLeft size={17} />
        </button>
      </div>
      <div className="intro-block">
        <div className="eyebrow">
          <span className="eyebrow-line" /> {t(($) => $.atlas.left.eyebrow)}
        </div>
        <h1>
          <Trans i18nKey={($) => $.atlas.left.tagline} components={{ br: <br />, em: <em /> }} />
        </h1>
        <p>
          {byDomain
            ? t(($) => $.atlas.left.domainsIntro, { count: fullModel.dimensionIds.length })
            : t(($) => $.atlas.left.dimensionsIntro, { count: DIMENSIONS.length })}
        </p>
      </div>
      <div className="nav-heading-row">
        <div className="nav-heading">{heading}</div>
        <GroupingSwitch value={viewModel.grouping} />
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
                <small>{t(($) => $.atlas.attributes, { count: node.attributeCount })}</small>
              </span>
            </button>
          );
        })}
      </nav>
      <div className="left-bottom">
        <div className="data-footnote">
          <span className="live-dot" /> {t(($) => $.review.list.count, { count: viewModel.facts.length })}
        </div>
      </div>
    </aside>
  );
}

const GROUPINGS: Grouping[] = ['dimension', 'domain'];

function GroupingSwitch({ value }: { value: Grouping }) {
  const { t } = useTranslation();
  const setGrouping = useProfileStore((state) => state.setGrouping);
  return (
    <div className="grouping-switch" role="group" aria-label={t(($) => $.atlas.left.grouping)}>
      {GROUPINGS.map((option) => (
        <button key={option} type="button" aria-pressed={value === option} onClick={() => setGrouping(option)}>
          {t(($) => $.atlas.left.groupings[option])}
        </button>
      ))}
    </div>
  );
}

import { useState, type CSSProperties } from 'react';
import type { LucideIcon } from 'lucide-react';
import { ChevronLeft, Compass, Fingerprint, Heart, Layers3, LayoutGrid, Orbit, Target } from 'lucide-react';
import { Trans, useTranslation } from 'react-i18next';
import { BrandMark } from '../BrandMark';
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

// In the order of the switch, left to right.
const GROUPINGS: { id: Grouping; Icon: LucideIcon }[] = [
  { id: 'dimension', Icon: Orbit },
  { id: 'domain', Icon: LayoutGrid },
];

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
  const setGrouping = useProfileStore((state) => state.setGrouping);
  const grouping = viewModel.grouping;
  const byDomain = grouping === 'domain';
  const heading = byDomain ? t(($) => $.atlas.left.domainsTitle) : t(($) => $.atlas.left.dimensionsTitle);
  // A switch brings the new list and intro in from the chosen option's side, until the list's last item is in.
  // The list the page opens with, and items a review view adds, just show.
  const [entering, setEntering] = useState(false);
  const side = GROUPINGS.findIndex((option) => option.id === grouping) ? 1 : -1;
  return (
    <aside
      className={`left-panel${open ? '' : ' is-collapsed'}`}
      aria-hidden={!open}
      onPointerDownCapture={onInteract}
      onFocusCapture={onInteract}
    >
      <div className="brand-row">
        <BrandMark className="brand-symbol" />
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
        <p key={grouping} className={`grouping-intro${entering ? ' is-entering' : ''}`}>
          {byDomain
            ? t(($) => $.atlas.left.domainsIntro, { count: fullModel.dimensionIds.length })
            : t(($) => $.atlas.left.dimensionsIntro, { count: DIMENSIONS.length })}
        </p>
      </div>
      <GroupingSwitch
        value={grouping}
        onChange={(next) => {
          if (next === grouping) return;
          setEntering(true);
          setGrouping(next);
        }}
      />
      <nav
        key={grouping}
        className={`dimension-list${byDomain ? ' is-compact' : ''}${entering ? ' is-entering' : ''}`}
        aria-label={heading}
        style={{ '--enter-side': side } as CSSProperties}
        onAnimationEnd={(event) => {
          if (event.target === event.currentTarget.lastElementChild) setEntering(false);
        }}
      >
        {viewModel.dimensionIds.map((id, index) => {
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
              style={{ '--dimension-color': color, '--enter-index': index } as CSSProperties}
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

/** Heads the list: what it and the first level of the map hold. The thumb slides under the chosen option. */
function GroupingSwitch({ value, onChange }: { value: Grouping; onChange: (grouping: Grouping) => void }) {
  const { t } = useTranslation();
  return (
    <div
      className="grouping-switch"
      role="group"
      aria-label={t(($) => $.atlas.left.grouping)}
      style={{ '--grouping-index': GROUPINGS.findIndex((option) => option.id === value) } as CSSProperties}
    >
      <span className="grouping-thumb" aria-hidden="true" />
      {GROUPINGS.map(({ id, Icon }) => (
        <button key={id} type="button" aria-pressed={value === id} onClick={() => onChange(id)}>
          <Icon size={15} strokeWidth={1.9} aria-hidden="true" />
          {t(($) => $.atlas.left.groupings[id])}
        </button>
      ))}
    </div>
  );
}

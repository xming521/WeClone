import { useMemo, useState, type CSSProperties } from 'react';
import type { LucideIcon } from 'lucide-react';
import { ChevronLeft, Compass, Fingerprint, Heart, Layers, LayoutGrid, Orbit, Target } from 'lucide-react';
import { Trans, useTranslation } from 'react-i18next';
import { BrandMark } from '../BrandMark';
import { collectFacts, DIMENSIONS, displayProfileText, type DimensionId } from '../data';
import type { Grouping, ReviewStatus } from '../data';
import { useProfileStore } from '../store';
import { scopeColor } from './scopeColor';

const DIMENSION_ICONS: Record<DimensionId, LucideIcon> = {
  1: Fingerprint,
  3: Target,
  4: Layers,
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
  // Review progress counts every fact, whatever view the review bar has filtered the map to.
  const review = useMemo(() => {
    const total = { pending: 0, approved: 0, rejected: 0 };
    fullModel.facts.forEach((fact) => total[fact.status]++);
    const perItem = new Map(
      fullModel.dimensionIds.map((id) => {
        const facts = collectFacts(fullModel, id);
        const count = (status: ReviewStatus) => facts.filter((fact) => fact.status === status).length;
        return [id, { facts: facts.length, pending: count('pending'), approved: count('approved') }] as const;
      }),
    );
    return { total, perItem };
  }, [fullModel]);
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
      <div className="wc-brand">
        <BrandMark className="wc-brand__mark" variant="small" />
        <span className="wc-brand__name">
          <b>WeClone</b>
          <small>Profile Atlas</small>
        </span>
        <button className="wc-iconbtn" type="button" aria-label={t(($) => $.atlas.left.close)} onClick={onClose}>
          <ChevronLeft size={16} strokeWidth={1.8} />
        </button>
      </div>
      <div className="intro-block">
        <div className="wc-eyebrow eyebrow">{t(($) => $.atlas.left.eyebrow)}</div>
        <h1 className="wc-hero">
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
        className={`wc-dims${byDomain ? ' is-compact' : ''}${entering ? ' is-entering' : ''}`}
        aria-label={heading}
        style={{ '--enter-side': side } as CSSProperties}
        onAnimationEnd={(event) => {
          if (event.target === event.currentTarget.lastElementChild) setEntering(false);
        }}
      >
        {viewModel.dimensionIds.map((id, index) => {
          const node = viewModel.nodes.get(id)!;
          const Icon = byDomain ? null : DIMENSION_ICONS[node.dim as DimensionId];
          const counts = review.perItem.get(id);
          const pending = counts?.pending ?? 0;
          const share = counts?.facts ? counts.approved / counts.facts : 0;
          const { style: colorVars, ...color } = scopeColor(node.dim);
          return (
            <button
              key={id}
              type="button"
              className="wc-dim"
              aria-current={focusedDimension === node.dim ? 'true' : undefined}
              {...color}
              title={node.description ? displayProfileText(node.description) : undefined}
              onClick={() => {
                reveal(id);
                onSelect();
              }}
              style={{ ...colorVars, '--enter-index': index } as CSSProperties}
            >
              <span className="wc-disc" data-progress aria-hidden="true" style={{ '--p': share } as CSSProperties}>
                {Icon ? (
                  <Icon size={17} strokeWidth={1.8} />
                ) : (
                  <span className="domain-initial">{[...node.name][0]}</span>
                )}
              </span>
              <span className="wc-dim__copy">
                <strong>{node.name}</strong>
                <small>{t(($) => $.atlas.attributes, { count: node.attributeCount })}</small>
              </span>
              {pending > 0 && (
                <span
                  className="wc-count wc-count--pending"
                  aria-label={t(($) => $.atlas.detail.pending, { count: pending })}
                >
                  {pending}
                </span>
              )}
            </button>
          );
        })}
      </nav>
      <div className="left-bottom">
        <div
          className="wc-progress"
          role="img"
          aria-label={t(($) => $.atlas.detail.progressCounts, {
            pending: review.total.pending,
            approved: review.total.approved,
            rejected: review.total.rejected,
          })}
        >
          {review.total.approved > 0 && <i data-k="adopted" style={{ flex: review.total.approved }} />}
          {review.total.pending > 0 && <i data-k="pending" style={{ flex: review.total.pending }} />}
          {review.total.rejected > 0 && <i data-k="rejected" style={{ flex: review.total.rejected }} />}
        </div>
        <div className="data-footnote">{t(($) => $.review.list.count, { count: fullModel.facts.length })}</div>
        <ul className="wc-legend">
          <li>
            <span className="wc-glyph" data-status="adopted" aria-hidden="true" />
            {t(($) => $.review.status.approved)} <b>{review.total.approved}</b>
          </li>
          <li>
            <span className="wc-glyph" data-status="pending" aria-hidden="true" />
            {t(($) => $.review.status.pending)} <b>{review.total.pending}</b>
          </li>
          <li>
            <span className="wc-glyph" data-status="rejected" aria-hidden="true" />
            {t(($) => $.review.status.rejected)} <b>{review.total.rejected}</b>
          </li>
        </ul>
      </div>
    </aside>
  );
}

/** Heads the list: what it and the first level of the map hold. The thumb slides under the chosen option. */
function GroupingSwitch({ value, onChange }: { value: Grouping; onChange: (grouping: Grouping) => void }) {
  const { t } = useTranslation();
  return (
    <div
      className="wc-seg wc-seg--fill wc-seg--lg wc-seg--slide grouping-switch"
      role="group"
      aria-label={t(($) => $.atlas.left.grouping)}
      style={
        {
          '--seg-i': GROUPINGS.findIndex((option) => option.id === value),
          '--seg-n': GROUPINGS.length,
        } as CSSProperties
      }
    >
      {GROUPINGS.map(({ id, Icon }) => (
        <button key={id} type="button" aria-pressed={value === id} onClick={() => onChange(id)}>
          <Icon size={16} strokeWidth={1.8} aria-hidden="true" />
          {t(($) => $.atlas.left.groupings[id])}
        </button>
      ))}
    </div>
  );
}

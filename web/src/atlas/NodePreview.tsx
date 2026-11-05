import { useLayoutEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ancestors, displayProfileText, type ProfileModel, type ProfileNode } from '../data';
import { Score } from '../review/ProfileScores';
import type { NodeMetrics } from './atlasMetrics';
import { scopeColor } from './scopeColor';

export function NodePreview({
  viewModel,
  node,
  metrics,
  x,
  y,
  width,
  height,
}: {
  viewModel: ProfileModel;
  node: ProfileNode;
  /** What the node's halo and line show: importance, confidence and chat support. */
  metrics: NodeMetrics;
  x: number;
  y: number;
  width: number;
  height: number;
}) {
  const { t } = useTranslation();
  const ref = useRef<HTMLDivElement>(null);
  const [place, setPlace] = useState({ left: x, top: y, tail: 'top-left' });
  const overview = node.kind === 'root' || node.kind === 'dimension';
  const facts = node.factIndices.map((index) => viewModel.facts[index]);
  const previews: string[] = [];
  function collect(current: ProfileNode) {
    for (const index of current.factIndices) {
      if (previews.length === 2) return;
      const fact = viewModel.facts[index];
      previews.push(t(($) => $.atlas.preview.fact, { attr: fact.attr, value: displayProfileText(fact.value) }));
    }
    for (const id of current.childIds) {
      if (previews.length === 2) return;
      collect(viewModel.nodes.get(id)!);
    }
  }
  if (overview) {
    previews.push(...node.childIds.slice(0, 3).map((id) => viewModel.nodes.get(id)!.name));
  } else if (node.kind === 'attribute') {
    previews.push(...facts.map((fact) => displayProfileText(fact.value)));
  } else {
    collect(node);
  }
  const path = ancestors(viewModel, node.id)
    .slice(1, -1)
    .map((parent) => parent.name)
    .join(' / ');
  const attributes = t(($) => $.atlas.attributes, { count: node.attributeCount });
  const count = node.childIds.length;
  const children =
    node.kind !== 'root'
      ? t(($) => $.atlas.preview.topics, { count })
      : viewModel.grouping === 'domain'
        ? t(($) => $.atlas.preview.domains, { count })
        : t(($) => $.atlas.preview.dimensions, { count });
  const summary =
    node.kind === 'attribute'
      ? t(($) => $.review.list.count, { count: facts.length })
      : overview
        ? `${children} · ${attributes}`
        : attributes;

  // The dot carries the colour the map paints the node's dimension or domain in.
  const dot =
    node.kind === 'root' ? { className: 'wc-dot node-preview-root' } : { className: 'wc-dot', ...scopeColor(node.dim) };

  useLayoutEffect(() => {
    const card = ref.current;
    if (!card) return;
    const left = Math.max(12, Math.min(x + 18, width - card.offsetWidth - 12));
    const above = y + 18 + card.offsetHeight > height - 12;
    const top = Math.max(12, Math.min(above ? y - card.offsetHeight - 18 : y + 18, height - card.offsetHeight - 12));
    // The tail dots sit on the corner of the card that faces the pointer.
    const tail = `${above ? 'bottom' : 'top'}-${x > left + card.offsetWidth / 2 ? 'right' : 'left'}`;
    setPlace({ left, top, tail });
  }, [x, y, width, height, node]);

  return (
    <div
      ref={ref}
      className={`graph-hover-tooltip wc-tip wc-tip--card${node.kind === 'attribute' ? ' node-preview-full' : ''}`}
      data-tail={place.tail}
      role="tooltip"
      style={{ left: place.left, top: place.top }}
    >
      <div className="wc-tip__title node-preview-heading">
        <span {...dot} aria-hidden="true" />
        <strong>{node.name}</strong>
      </div>
      {path && <div className="wc-tip__meta node-preview-path">{path}</div>}
      <div className="wc-tip__meta">{summary}</div>
      <div className="node-preview-scores">
        <Score kind="importance" value={metrics.importance ?? undefined} compact />
        <Score kind="confidence" value={metrics.confidence ?? undefined} compact />
        <span>{t(($) => $.scores.support, { count: metrics.support })}</span>
      </div>
      <div className="node-preview-content">
        {overview ? <p>{previews.join(' · ')}</p> : previews.map((preview, index) => <p key={index}>{preview}</p>)}
      </div>
    </div>
  );
}

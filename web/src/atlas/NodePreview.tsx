import { useLayoutEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  ancestors,
  dimensionColor,
  displayProfileText,
  ROOT_COLOR,
  type ProfileModel,
  type ProfileNode,
} from '../data';
import { Score } from '../review/ProfileScores';
import type { NodeMetrics } from './atlasMetrics';

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
  const [position, setPosition] = useState({ left: x, top: y });
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

  useLayoutEffect(() => {
    const card = ref.current;
    if (!card) return;
    const left = Math.max(12, Math.min(x + 18, width - card.offsetWidth - 12));
    const top = Math.max(
      12,
      Math.min(
        y + 18 + card.offsetHeight > height - 12 ? y - card.offsetHeight - 18 : y + 18,
        height - card.offsetHeight - 12,
      ),
    );
    setPosition({ left, top });
  }, [x, y, width, height, node]);

  return (
    <div
      ref={ref}
      className={`graph-hover-tooltip${node.kind === 'attribute' ? ' node-preview-full' : ''}`}
      role="tooltip"
      style={position}
    >
      <div className="node-preview-heading">
        <span style={{ background: node.kind === 'root' ? ROOT_COLOR : dimensionColor(node.dim) }} />
        <strong>{node.name}</strong>
      </div>
      {path && <div className="node-preview-path">{path}</div>}
      <div className="node-preview-summary">{summary}</div>
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

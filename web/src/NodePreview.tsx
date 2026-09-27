import { useLayoutEffect, useRef, useState } from 'react';
import { ancestors, DIMENSION_COLORS, type ProfileModel, type ProfileNode } from './data';

export function NodePreview({ model, node, x, y, width, height }: {
  model: ProfileModel;
  node: ProfileNode;
  x: number;
  y: number;
  width: number;
  height: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState({ left: x, top: y });
  const overview = node.kind === 'root' || node.kind === 'dimension';
  const facts = node.factIndices.map((index) => model.facts[index]);
  const previews: string[] = [];
  function collect(current: ProfileNode) {
    for (const index of current.factIndices) {
      if (previews.length === 2) return;
      const fact = model.facts[index];
      previews.push(`${fact.attr}：${fact.value}`);
    }
    for (const id of current.childIds) {
      if (previews.length === 2) return;
      collect(model.nodes.get(id)!);
    }
  }
  if (overview) {
    previews.push(...node.childIds.slice(0, 3).map((id) => model.nodes.get(id)!.name));
  } else if (node.kind === 'attribute') {
    previews.push(...facts.map((fact) => fact.value));
  } else {
    collect(node);
  }
  const path = ancestors(model, node.id).slice(1, -1).map((parent) => parent.name).join(' / ');
  const summary = node.kind === 'attribute'
    ? `${facts.length} 条画像 · ${new Set(facts.flatMap((fact) => fact.source_ids)).size} 条来源`
    : `${overview ? `${node.childIds.length} 个${node.kind === 'root' ? '维度' : '主题'} · ` : ''}${node.attributeCount} 项画像属性`;

  useLayoutEffect(() => {
    const card = ref.current;
    if (!card) return;
    const left = Math.max(12, Math.min(x + 18, width - card.offsetWidth - 12));
    const top = Math.max(12, Math.min(y + 18 + card.offsetHeight > height - 12 ? y - card.offsetHeight - 18 : y + 18, height - card.offsetHeight - 12));
    setPosition({ left, top });
  }, [x, y, width, height, node]);

  return <div ref={ref} className={`graph-hover-tooltip${node.kind === 'attribute' ? ' node-preview-full' : ''}`} role="tooltip" style={position}>
    <div className="node-preview-heading"><span style={{ background: node.kind === 'root' ? '#59616d' : DIMENSION_COLORS[node.dim ?? 1] }} /><strong>{node.name}</strong></div>
    {path && <div className="node-preview-path">{path}</div>}
    <div className="node-preview-summary">{summary}</div>
    <div className="node-preview-content">
      {overview ? <p>{previews.join(' · ')}</p> : previews.map((preview, index) => <p key={index}>{preview}</p>)}
    </div>
  </div>;
}

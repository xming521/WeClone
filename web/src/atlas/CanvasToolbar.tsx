import type { RefObject } from 'react';
import type { Graph } from '@antv/g6';
import * as Tooltip from '@radix-ui/react-tooltip';
import { Maximize2, Minus, Plus, RotateCcw } from 'lucide-react';
import { levelIsExpanded } from '../data';
import { useProfileStore } from '../store';
import { IconButton } from './IconButton';

export function CanvasToolbar({ graphRef }: { graphRef: RefObject<Graph | null> }) {
  const viewModel = useProfileStore((state) => state.viewModel)!;
  const focusedDimension = useProfileStore((state) => state.focusedDimension);
  const expandedIds = useProfileStore((state) => state.expandedIds);
  const toggleLevel = useProfileStore((state) => state.toggleLevel);
  const reset = useProfileStore((state) => state.reset);
  return (
    <div className="canvas-toolbar">
      <div className="level-controls">
        {([1, 2, 3] as const).map((level) => {
          const expanded = levelIsExpanded(viewModel, expandedIds, level, focusedDimension);
          const action =
            focusedDimension === null ? '先点击一个维度' : `${expanded ? '收起' : '展开'}当前维度的${level}级节点`;
          return (
            <Tooltip.Root key={level} delayDuration={250}>
              <Tooltip.Trigger asChild>
                <button
                  className={`level-button${expanded ? ' is-active' : ''}${focusedDimension === null ? ' is-disabled' : ''}`}
                  type="button"
                  aria-label={action}
                  aria-disabled={focusedDimension === null}
                  aria-pressed={expanded}
                  onClick={() => toggleLevel(level)}
                >
                  {level}级
                </button>
              </Tooltip.Trigger>
              <Tooltip.Portal>
                <Tooltip.Content className="tooltip" sideOffset={8}>
                  {action}
                  <Tooltip.Arrow className="tooltip-arrow" />
                </Tooltip.Content>
              </Tooltip.Portal>
            </Tooltip.Root>
          );
        })}
      </div>
      <span className="toolbar-separator" />
      <div className="view-controls">
        <IconButton label="放大" onClick={() => void graphRef.current?.zoomBy(1.2, { duration: 250 })}>
          <Plus size={17} />
        </IconButton>
        <IconButton label="缩小" onClick={() => void graphRef.current?.zoomBy(0.83, { duration: 250 })}>
          <Minus size={17} />
        </IconButton>
        <span className="toolbar-separator" />
        <IconButton label="适应画布" onClick={() => void graphRef.current?.fitView()}>
          <Maximize2 size={16} />
        </IconButton>
        <span className="toolbar-separator" />
        <button className="reset-button" type="button" onClick={() => reset()}>
          <RotateCcw size={15} />
          复原
        </button>
      </div>
    </div>
  );
}

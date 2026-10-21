import type { RefObject } from 'react';
import * as Tooltip from '@radix-ui/react-tooltip';
import { Maximize2, Minus, MoonStar, Plus, RotateCcw } from 'lucide-react';
import { levelIsExpanded } from '../data';
import { useProfileStore } from '../store';
import type { AtlasControls } from './atlasEngine';
import { IconButton } from './IconButton';
import { nightIntroAllowed, replayNightIntro } from './nightIntro';

export function CanvasToolbar({ atlasRef }: { atlasRef: RefObject<AtlasControls | null> }) {
  const viewModel = useProfileStore((state) => state.viewModel)!;
  const focusedDimension = useProfileStore((state) => state.focusedDimension);
  const expandedIds = useProfileStore((state) => state.expandedIds);
  const toggleLevel = useProfileStore((state) => state.toggleLevel);
  const reset = useProfileStore((state) => state.reset);
  const byDomain = viewModel.grouping === 'domain';
  const level1 = byDomain ? '领域' : '维度';
  // Domains hold topics, which hold attributes: there is no third level of groups.
  const levels = byDomain ? ([1, 2] as const) : ([1, 2, 3] as const);
  return (
    <div className="canvas-toolbar">
      <div className="level-controls">
        {levels.map((level) => {
          const expanded = levelIsExpanded(viewModel, expandedIds, level, focusedDimension);
          const action =
            focusedDimension === null
              ? `先点击一个${level1}`
              : `${expanded ? '收起' : '展开'}当前${level1}的${level}级节点`;
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
        <IconButton label="放大" onClick={() => atlasRef.current?.zoomBy(1.2)}>
          <Plus size={17} />
        </IconButton>
        <IconButton label="缩小" onClick={() => atlasRef.current?.zoomBy(0.83)}>
          <Minus size={17} />
        </IconButton>
        <span className="toolbar-separator" />
        <IconButton label="适应画布" onClick={() => atlasRef.current?.fit()}>
          <Maximize2 size={16} />
        </IconButton>
        <span className="toolbar-separator" />
        <button className="reset-button" type="button" onClick={() => reset()}>
          <RotateCcw size={15} />
          复原
        </button>
        {/* The intro tells the five dimensions apart by colour. */}
        {nightIntroAllowed() && !byDomain && (
          // The map starts over as on a first visit, and the intro plays over it.
          <IconButton
            label="重放开场动画"
            onClick={() => {
              replayNightIntro();
              reset();
            }}
          >
            <MoonStar size={16} />
          </IconButton>
        )}
      </div>
    </div>
  );
}

import type { RefObject } from 'react';
import * as Tooltip from '@radix-ui/react-tooltip';
import { Maximize2, Minus, MoonStar, Plus, RotateCcw } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { levelIsExpanded } from '../data';
import { useProfileStore } from '../store';
import type { AtlasControls } from './atlasEngine';
import { IconButton } from './IconButton';
import { nightIntroAllowed, replayNightIntro } from './nightIntro';

export function CanvasToolbar({ atlasRef }: { atlasRef: RefObject<AtlasControls | null> }) {
  const { t } = useTranslation();
  const viewModel = useProfileStore((state) => state.viewModel)!;
  const focusedDimension = useProfileStore((state) => state.focusedDimension);
  const expandedIds = useProfileStore((state) => state.expandedIds);
  const toggleLevel = useProfileStore((state) => state.toggleLevel);
  const reset = useProfileStore((state) => state.reset);
  const byDomain = viewModel.grouping === 'domain';
  // Domains hold topics, which hold attributes: there is no third level of groups.
  const levels = byDomain ? ([1, 2] as const) : ([1, 2, 3] as const);
  return (
    <div className="canvas-toolbar">
      <div className="level-controls">
        {levels.map((level) => {
          const expanded = levelIsExpanded(viewModel, expandedIds, level, focusedDimension);
          const action =
            focusedDimension === null
              ? byDomain
                ? t(($) => $.atlas.toolbar.pickDomain)
                : t(($) => $.atlas.toolbar.pickDimension)
              : byDomain
                ? expanded
                  ? t(($) => $.atlas.toolbar.collapseDomain, { level })
                  : t(($) => $.atlas.toolbar.expandDomain, { level })
                : expanded
                  ? t(($) => $.atlas.toolbar.collapseDimension, { level })
                  : t(($) => $.atlas.toolbar.expandDimension, { level });
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
                  {t(($) => $.atlas.toolbar.level, { level })}
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
        <IconButton label={t(($) => $.atlas.toolbar.zoomIn)} onClick={() => atlasRef.current?.zoomBy(1.2)}>
          <Plus size={17} />
        </IconButton>
        <IconButton label={t(($) => $.atlas.toolbar.zoomOut)} onClick={() => atlasRef.current?.zoomBy(0.83)}>
          <Minus size={17} />
        </IconButton>
        <span className="toolbar-separator" />
        <IconButton label={t(($) => $.atlas.toolbar.fit)} onClick={() => atlasRef.current?.fit()}>
          <Maximize2 size={16} />
        </IconButton>
        <span className="toolbar-separator" />
        <button className="reset-button" type="button" onClick={() => reset()}>
          <RotateCcw size={15} />
          {t(($) => $.atlas.toolbar.reset)}
        </button>
        {/* The intro tells the five dimensions apart by colour. */}
        {nightIntroAllowed() && !byDomain && (
          // The map starts over as on a first visit, and the intro plays over it.
          <IconButton
            label={t(($) => $.atlas.toolbar.replayIntro)}
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

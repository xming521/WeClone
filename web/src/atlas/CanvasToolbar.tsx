import type { RefObject } from 'react';
import { ImageDown, Maximize2, Minus, MoonStar, Plus, RotateCcw } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { levelIsExpanded } from '../data';
import { toast } from '../feedback/toast';
import { useProfileStore } from '../store';
import type { AtlasControls } from './atlasEngine';
import { IconButton, Tip } from './IconButton';
import { nightIntroAllowed, replayNightIntro } from './nightIntro';

export function CanvasToolbar({ atlasRef }: { atlasRef: RefObject<AtlasControls | null> }) {
  const { t } = useTranslation();
  const viewModel = useProfileStore((state) => state.viewModel)!;
  const focusedDimension = useProfileStore((state) => state.focusedDimension);
  const expandedIds = useProfileStore((state) => state.expandedIds);
  const toggleLevel = useProfileStore((state) => state.toggleLevel);
  const reset = useProfileStore((state) => state.reset);
  const byDomain = viewModel.grouping === 'domain';
  // The intro tells the five dimensions apart by colour.
  const replayShown = nightIntroAllowed() && !byDomain;
  // The map as it shows now, to share: what is expanded, the zoom and the colour scheme.
  async function saveImage() {
    const blob = await atlasRef.current?.snapshot();
    if (!blob) return;
    const file = 'weclone-map.png';
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = file;
    document.body.append(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    toast(t(($) => $.toast.downloaded, { file }));
  }
  // Domains hold topics, which hold attributes: there is no third level of groups.
  const levels = byDomain ? ([1, 2] as const) : ([1, 2, 3] as const);
  return (
    <div className="canvas-toolbar wc-toolbar">
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
            <Tip key={level} content={action}>
              <button
                className={`wc-level${focusedDimension === null ? ' is-disabled' : ''}`}
                type="button"
                aria-label={action}
                aria-disabled={focusedDimension === null}
                aria-pressed={expanded}
                onClick={() => toggleLevel(level)}
              >
                <i className={`wc-i wc-i-orbit-${level}`} aria-hidden="true" />
                {t(($) => $.atlas.toolbar.level, { level })}
              </button>
            </Tip>
          );
        })}
      </div>
      <span className="wc-sep" aria-hidden="true" />
      <div className="view-controls">
        <IconButton label={t(($) => $.atlas.toolbar.zoomIn)} onClick={() => atlasRef.current?.zoomBy(1.2)}>
          <Plus size={16} strokeWidth={1.8} aria-hidden="true" />
        </IconButton>
        <IconButton label={t(($) => $.atlas.toolbar.zoomOut)} onClick={() => atlasRef.current?.zoomBy(0.83)}>
          <Minus size={16} strokeWidth={1.8} aria-hidden="true" />
        </IconButton>
        <IconButton label={t(($) => $.atlas.toolbar.fit)} onClick={() => atlasRef.current?.fit()}>
          <Maximize2 size={16} strokeWidth={1.8} aria-hidden="true" />
        </IconButton>
        <IconButton label={t(($) => $.atlas.toolbar.saveImage)} onClick={() => void saveImage()}>
          <ImageDown size={16} strokeWidth={1.8} aria-hidden="true" />
        </IconButton>
        <span className="wc-sep" aria-hidden="true" />
        <IconButton
          label={t(($) => $.atlas.toolbar.reset)}
          align={replayShown ? 'start' : 'end'}
          onClick={() => reset()}
        >
          <RotateCcw size={16} strokeWidth={1.8} aria-hidden="true" />
        </IconButton>
        {replayShown && (
          // The map starts over as on a first visit, and the intro plays over it.
          <IconButton
            label={t(($) => $.atlas.toolbar.replayIntro)}
            align="end"
            onClick={() => {
              replayNightIntro();
              reset();
            }}
          >
            <MoonStar size={16} strokeWidth={1.8} aria-hidden="true" />
          </IconButton>
        )}
      </div>
    </div>
  );
}

import { useEffect, useRef, useState } from 'react';
import * as Tooltip from '@radix-ui/react-tooltip';
import { Menu, Move } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { AtlasControls } from './atlas/atlasEngine';
import { CanvasToolbar } from './atlas/CanvasToolbar';
import { DetailPanel } from './atlas/DetailPanel';
import { GraphCanvas } from './atlas/GraphCanvas';
import { LeftPanel } from './atlas/LeftPanel';
import { SearchBox } from './atlas/SearchBox';
import { ProfileActions } from './review/ProfileActions';
import { ReviewBar } from './review/ReviewBar';
import { ReviewOverlays } from './review/ReviewOverlays';
import { DIMENSIONS } from './data';
import { useProfileStore } from './store';
import { AuthGate, LogoutButton } from './Auth';
import { LanguageSwitch } from './LanguageSwitch';
import { ModeSwitch, modeFromHash } from './ModeSwitch';
import { StyleApp } from './style/StyleApp';
import { AssistantApp } from './assistant/AssistantApp';
import { BotsApp } from './bots/BotsApp';
import { SummaryApp } from './summary/SummaryApp';

function Atlas() {
  const { t } = useTranslation();
  const viewModel = useProfileStore((state) => state.viewModel)!;
  const domainCount = useProfileStore((state) => state.fullModel!.dimensionIds.length);
  const atlasRef = useRef<AtlasControls | null>(null);
  const [leftOpen, setLeftOpen] = useState(true);
  const [leftTouched, setLeftTouched] = useState(false);
  const focusedDimension = useProfileStore((state) => state.focusedDimension);
  const resetVersion = useProfileStore((state) => state.resetVersion);
  const detailOpen = useProfileStore((state) => state.detailOpen);
  const closeDetail = useProfileStore((state) => state.closeDetail);
  const active = viewModel.nodes.get(`dimension:${focusedDimension}`);
  const byDomain = viewModel.grouping === 'domain';

  useEffect(() => {
    if (leftTouched) return;
    const timer = window.setTimeout(() => setLeftOpen(false), 5000);
    return () => window.clearTimeout(timer);
  }, [leftTouched]);

  return (
    <Tooltip.Provider>
      <div className={`app-shell${leftOpen ? ' left-open' : ''}${detailOpen ? ' detail-open' : ''}`}>
        <LeftPanel
          open={leftOpen}
          onClose={() => {
            setLeftTouched(true);
            setLeftOpen(false);
          }}
          onInteract={() => setLeftTouched(true)}
          onSelect={() => {
            setLeftTouched(true);
            setLeftOpen(false);
          }}
        />
        <main className="atlas-main">
          <div className="canvas-atmosphere" />
          <div className="topbar">
            <div className="topbar-leading">
              {!leftOpen && (
                <button
                  className="sidebar-open"
                  type="button"
                  aria-label={t(($) => $.app.openLeftPanel)}
                  onClick={() => {
                    setLeftTouched(true);
                    setLeftOpen(true);
                  }}
                >
                  <Menu size={17} />
                  <span>{byDomain ? t(($) => $.atlas.left.domainsTitle) : t(($) => $.atlas.left.dimensionsTitle)}</span>
                </button>
              )}
              <ModeSwitch mode="profile" />
              <div className="topbar-heading">
                <span className="topbar-divider">/</span>{' '}
                <strong>
                  {active?.name ??
                    (byDomain
                      ? t(($) => $.app.domainOverview, { count: domainCount })
                      : t(($) => $.app.dimensionOverview, { count: DIMENSIONS.length }))}
                </strong>
              </div>
            </div>
            <div className="topbar-actions">
              <SearchBox />
              <ProfileActions />
              <LanguageSwitch />
              <LogoutButton />
            </div>
          </div>
          <ReviewBar />
          {!viewModel.facts.length && <div className="graph-empty">{t(($) => $.app.emptyView)}</div>}
          <GraphCanvas key={resetVersion} atlasRef={atlasRef} />
          <CanvasToolbar atlasRef={atlasRef} />
          <div className="canvas-help">
            <Move size={15} /> {t(($) => $.app.canvasHelp)}
          </div>
        </main>
        <ReviewOverlays />
        <DetailPanel open={detailOpen} onClose={closeDetail} />
      </div>
    </Tooltip.Provider>
  );
}

function ProfileApp() {
  const { t } = useTranslation();
  const loaded = useProfileStore((state) => state.viewModel !== null);
  const error = useProfileStore((state) => state.error);
  const load = useProfileStore((state) => state.load);
  const reload = useProfileStore((state) => state.reload);
  useEffect(() => {
    void load();
  }, [load]);
  if (!loaded && error)
    return (
      <div className="load-state">
        <span>✳</span>
        <h1>{t(($) => $.app.loadFailed)}</h1>
        <p>{error}</p>
        <button onClick={() => void reload()}>{t(($) => $.common.retry)}</button>
      </div>
    );
  if (!loaded)
    return (
      <div className="load-state">
        <span className="loading-mark">✳</span>
        <h1>{t(($) => $.app.loading)}</h1>
      </div>
    );
  return <Atlas />;
}

function useMode() {
  const [mode, setMode] = useState(() => modeFromHash(window.location.hash));
  useEffect(() => {
    const update = () => setMode(modeFromHash(window.location.hash));
    window.addEventListener('hashchange', update);
    return () => window.removeEventListener('hashchange', update);
  }, []);
  return mode;
}

export default function App() {
  const mode = useMode();
  const Page = { profile: ProfileApp, style: StyleApp, summary: SummaryApp, assistant: AssistantApp, bots: BotsApp }[
    mode
  ];
  return (
    <AuthGate>
      <Page />
    </AuthGate>
  );
}

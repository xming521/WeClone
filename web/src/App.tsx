import { useEffect, useRef, useState } from 'react';
import * as Tooltip from '@radix-ui/react-tooltip';
import { Menu, Move } from 'lucide-react';
import type { AtlasControls } from './atlas/atlasEngine';
import { CanvasToolbar } from './atlas/CanvasToolbar';
import { DetailPanel } from './atlas/DetailPanel';
import { GraphCanvas } from './atlas/GraphCanvas';
import { LeftPanel } from './atlas/LeftPanel';
import { SearchBox } from './atlas/SearchBox';
import { ProfileActions } from './review/ProfileActions';
import { ReviewBar } from './review/ReviewBar';
import { ReviewOverlays } from './review/ReviewOverlays';
import { DIMENSIONS, chineseNumeral } from './data';
import { useProfileStore } from './store';
import { AuthGate, LogoutButton } from './Auth';
import { ModeSwitch, modeFromHash } from './ModeSwitch';
import { StyleApp } from './style/StyleApp';
import { AssistantApp } from './assistant/AssistantApp';
import { SummaryApp } from './summary/SummaryApp';

function Atlas() {
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
                  aria-label="展开左栏"
                  onClick={() => {
                    setLeftTouched(true);
                    setLeftOpen(true);
                  }}
                >
                  <Menu size={17} />
                  <span>{byDomain ? '画像领域' : '画像维度'}</span>
                </button>
              )}
              <ModeSwitch mode="profile" />
              <div className="topbar-heading">
                <span className="topbar-divider">/</span>{' '}
                <strong>
                  {active?.name ??
                    (byDomain
                      ? `${chineseNumeral(domainCount)}个领域总览`
                      : `${chineseNumeral(DIMENSIONS.length)}维总览`)}
                </strong>
              </div>
            </div>
            <div className="topbar-actions">
              <SearchBox />
              <ProfileActions />
              <LogoutButton />
            </div>
          </div>
          <ReviewBar />
          {!viewModel.facts.length && <div className="graph-empty">当前视图没有画像记录</div>}
          <GraphCanvas key={resetVersion} atlasRef={atlasRef} />
          <CanvasToolbar atlasRef={atlasRef} />
          <div className="canvas-help">
            <Move size={15} /> 悬停点亮归属路径 · 点击展开并聚焦，再点一次收起 · 拖动节点牵动图谱
          </div>
        </main>
        <ReviewOverlays />
        <DetailPanel open={detailOpen} onClose={closeDetail} />
      </div>
    </Tooltip.Provider>
  );
}

function ProfileApp() {
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
        <h1>画像地图暂时无法打开</h1>
        <p>{error}</p>
        <button onClick={() => void reload()}>重试</button>
      </div>
    );
  if (!loaded)
    return (
      <div className="load-state">
        <span className="loading-mark">✳</span>
        <h1>正在展开画像地图</h1>
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
  const Page = { profile: ProfileApp, style: StyleApp, summary: SummaryApp, assistant: AssistantApp }[mode];
  return (
    <AuthGate>
      <Page />
    </AuthGate>
  );
}

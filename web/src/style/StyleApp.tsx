import { useEffect, useMemo, useState, type CSSProperties } from 'react';
import type { LucideIcon } from 'lucide-react';
import { Keyboard, MessagesSquare, Smile, Type } from 'lucide-react';
import { api } from '../api';
import { LogoutButton } from '../Auth';
import { ModeSwitch } from '../ModeSwitch';
import { BubbleField } from './BubbleField';
import { TraitDetail } from './TraitDetail';
import { MIN_RATE, TIERS, TRAIT_GROUPS, selectTraits, type TraitGroupId } from './traits';
import type { StyleInput } from './types';
import './styleView.css';

const GROUP_ICONS: Record<TraitGroupId, LucideIcon> = {
  typing: Keyboard,
  words: Type,
  emotion: Smile,
  manner: MessagesSquare,
};

export function StyleApp() {
  const [data, setData] = useState<StyleInput | null>(null);
  const [error, setError] = useState('');
  const [selected, setSelected] = useState<string | null>(null);
  const [focus, setFocus] = useState<TraitGroupId | null>(null);

  useEffect(() => {
    let live = true;
    api
      .get<StyleInput>('style')
      .then((value) => {
        if (live) setData(value);
      })
      .catch((reason: Error) => {
        if (live) setError(reason.message);
      });
    return () => {
      live = false;
    };
  }, []);

  const traits = useMemo(() => (data ? selectTraits(data) : []), [data]);

  useEffect(() => {
    const close = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setSelected(null);
    };
    window.addEventListener('keydown', close);
    return () => window.removeEventListener('keydown', close);
  }, []);

  if (error)
    return (
      <div className="load-state">
        <span>✳</span>
        <h1>风格结果暂时无法打开</h1>
        <p>{error}</p>
        <a className="style-back" href="#/">
          回到人生画像
        </a>
      </div>
    );
  if (!data)
    return (
      <div className="load-state">
        <span className="loading-mark">✳</span>
        <h1>正在读取聊天风格</h1>
      </div>
    );

  const trait = traits.find((entry) => entry.key === selected) ?? null;
  const focusGroup = TRAIT_GROUPS.find((group) => group.id === focus);
  const { start, end } = data.run.window;
  const messages = data.run.scope.surface?.messages;

  return (
    <div className={`app-shell left-open style-shell${trait ? ' detail-open' : ''}`}>
      <aside className="left-panel">
        <div className="brand-row">
          <span className="brand-symbol">✳</span>
          <span>
            WeClone<span className="brand-separator">/</span>Style
          </span>
        </div>
        <div className="intro-block">
          <div className="eyebrow">
            <span className="eyebrow-line" /> HOW YOU TALK
          </div>
          <h1>
            你怎么聊天，
            <br />
            <em>一眼就能看出来。</em>
          </h1>
          {traits.length > 0 && (
            <p>
              最明显的是
              {traits
                .slice(0, 3)
                .map((entry) => `「${entry.title}」`)
                .join('')}
              。
            </p>
          )}
        </div>
        <div className="nav-heading">风格侧面</div>
        <nav className="dimension-list" aria-label="风格侧面">
          {TRAIT_GROUPS.map((group) => {
            const Icon = GROUP_ICONS[group.id];
            const count = traits.filter((entry) => entry.group === group.id).length;
            return (
              <button
                key={group.id}
                type="button"
                className={`dimension-item ${focus === group.id ? 'active' : ''}`}
                onClick={() => setFocus(focus === group.id ? null : group.id)}
                style={{ '--dimension-color': group.color } as CSSProperties}
              >
                <span className="dimension-icon">
                  <Icon size={19} strokeWidth={1.7} />
                </span>
                <span className="dimension-copy">
                  <strong>{group.name}</strong>
                  <small>
                    {count} 个特点 · {group.hint}
                  </small>
                </span>
              </button>
            );
          })}
        </nav>
        <div className="left-bottom">
          <div className="data-footnote">
            <span className="live-dot" /> {start?.slice(0, 7)} 至 {end?.slice(0, 7)} · {messages} 条消息
          </div>
        </div>
      </aside>

      <main className="atlas-main">
        <div className="canvas-atmosphere" />
        <div className="topbar">
          <div className="topbar-leading">
            <ModeSwitch mode="style" />
            <div className="topbar-heading">
              <span className="topbar-divider">/</span> <strong>{focusGroup?.name ?? '主要特点'}</strong>
            </div>
          </div>
          <div className="topbar-actions">
            <LogoutButton />
          </div>
        </div>
        {traits.length ? (
          <BubbleField traits={traits} selected={selected} focus={focus} onSelect={setSelected} />
        ) : (
          <div className="graph-empty">还没有足够明显的风格特点</div>
        )}
        <div className="canvas-help bubble-legend">
          {TIERS.map((tier) => (
            <span key={tier.id}>
              <i className={`legend-swatch tier-${tier.id}`} />
              {tier.label}
            </span>
          ))}
          <span className="legend-note">
            泡泡越大越常见 · 只列出每 {Math.round(1 / MIN_RATE)} 条至少出现 1 次的特点 · 点击看详情
          </span>
        </div>
      </main>

      <TraitDetail trait={trait} onClose={() => setSelected(null)} />
    </div>
  );
}

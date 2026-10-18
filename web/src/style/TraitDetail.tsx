import { useState, type CSSProperties } from 'react';
import { Sparkles, X } from 'lucide-react';
import { TRAIT_GROUPS, VARIATION_TEXT, formatPercent, frequencyText, tierOf, type Trait } from './traits';

export function TraitDetail({ trait: open, onClose }: { trait: Trait | null; onClose: () => void }) {
  // Keep the last trait on screen while the panel slides out.
  const [shown, setShown] = useState(open);
  if (open && open !== shown) setShown(open);
  const trait = open ?? shown;
  const group = TRAIT_GROUPS.find((entry) => entry.id === trait?.group);
  const tier = trait ? tierOf(trait.rate) : null;
  return (
    <aside
      className={`detail-panel trait-detail${open ? ' is-open' : ''}`}
      aria-hidden={!open}
      style={{ '--bubble-color': group?.color } as CSSProperties}
    >
      <button className="detail-close" type="button" aria-label="关闭右侧栏" onClick={onClose}>
        <X size={17} />
      </button>
      <div className="detail-topline">
        <span className="detail-kicker">
          <Sparkles size={14} /> 风格特点
        </span>
        <span className="detail-serial">{group?.name}</span>
      </div>
      {trait && (
        <div className="detail-content">
          <div className="trait-hero">
            {/* The opened bubble again, popping in each time another one is chosen. */}
            <div key={trait.key} className={`trait-emblem tier-${tier?.id}`}>
              <span className="trait-bubble-body">
                <strong>{formatPercent(trait.rate)}</strong>
              </span>
            </div>
            <div>
              <h2 className="trait-title">{trait.title}</h2>
              <p className="trait-note">{trait.note}</p>
            </div>
          </div>
          <p className="trait-frequency">
            <b>{tier?.label}</b>
            {trait.when ? `${trait.when}，` : ''}
            {frequencyText(trait.rate)}
          </p>

          {trait.examples.length > 0 && (
            <section className="trait-section">
              <h3>常见写法</h3>
              <div className="trait-examples">
                {trait.examples.map((example) => (
                  <span key={example}>{example}</span>
                ))}
              </div>
            </section>
          )}

          {trait.contacts.length > 1 && (
            <section className="trait-section">
              <h3>跟谁聊的时候</h3>
              {trait.variation && VARIATION_TEXT[trait.variation] && (
                <p className="trait-variation">{VARIATION_TEXT[trait.variation]}</p>
              )}
              <div className="trait-contacts">
                {trait.contacts.map(({ contact, rate }) => (
                  <div key={contact.id} className="trait-contact">
                    <span className="trait-contact-name" title={contact.label}>
                      {contact.label}
                    </span>
                    <span className="trait-contact-bar">
                      <i style={{ width: `${rate * 100}%` }} />
                      <b style={{ left: `${trait.rate * 100}%` }} />
                    </span>
                    <span className="trait-contact-value">{formatPercent(rate)}</span>
                  </div>
                ))}
              </div>
              <p className="trait-source">竖线是整体水平。各人的数字已按消息长短校正。</p>
            </section>
          )}

          <p className="trait-source">
            {trait.sampled
              ? `依据：抽样的 ${trait.messages} ${trait.unit ?? '条消息'}，由模型判断`
              : `依据：全部 ${trait.messages} ${trait.unit ?? '条消息'}`}
          </p>
        </div>
      )}
    </aside>
  );
}

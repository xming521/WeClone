import { ArrowUpRight, BookOpen, Bug, Github, Globe, MessageCircle, Send, Users } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { Trans, useTranslation } from 'react-i18next';
import { BrandMark } from '../BrandMark';
import { useRun } from './useRun';
import { PageTopbar } from './PageTopbar';
import '../summary/summary.css';
import '../assistant/assistant.css';
import './settings.css';

// The project's public addresses, as README.md and README_zh.md list them; `zh`/`en` pick the page in that language.
const LINKS = {
  home: 'https://www.weclone.love/',
  docs: {
    zh: 'https://docs.weclone.love/zh/docs/introduce/what-is-weclone',
    en: 'https://docs.weclone.love/en/docs/introduce/what-is-weclone',
  },
  repo: 'https://github.com/xming521/WeClone',
  issues: 'https://github.com/xming521/WeClone/issues',
  faq: { zh: 'https://docs.weclone.love/zh/docs/introduce/FAQ', en: 'https://docs.weclone.love/en/docs/introduce/FAQ' },
  qq: 'https://qm.qq.com/cgi-bin/qm/qr?k=wNdgbOVT6oFOJ2wlMLsolUXErW9ESLpk&jump_from=webapi&authKey=z/reOp6YLyvR4Tl2k2nYMsLoMC3w9/99ucgKMX0oRGlxDV/WbYnvq2QxODoIkfxn',
  telegram: 'https://t.me/+JEdak4m0XEQ3NGNl',
  x: 'https://x.com/weclone567',
  xiaohongshu: 'https://www.xiaohongshu.com/user/profile/628109730000000021029de4',
  license: 'https://github.com/xming521/WeClone/blob/master/LICENSE',
} as const;

// The open-source projects README.md thanks.
const THANKS = [
  ['LLaMA-Factory', 'https://github.com/hiyouga/LLaMA-Factory'],
  ['AstrBot', 'https://github.com/AstrBotDevs/AstrBot'],
  ['LangBot', 'https://github.com/RockChinQ/LangBot'],
] as const;

function LinkCard({ href, icon: Icon, title, text }: { href: string; icon: LucideIcon; title: string; text: string }) {
  return (
    <a className="about-link" href={href} target="_blank" rel="noreferrer">
      <Icon size={18} strokeWidth={1.8} aria-hidden="true" />
      <span>
        <strong>{title}</strong>
        <span>{text}</span>
      </span>
      <ArrowUpRight size={15} strokeWidth={1.8} aria-hidden="true" />
    </a>
  );
}

export function AboutApp() {
  const { t, i18n } = useTranslation();
  const zh = i18n.language === 'zh-CN';
  const version = useRun()?.version;
  return (
    <div className="summary-page">
      <PageTopbar mode="about" heading={t(($) => $.about.title)} />
      <main className="summary-main assistant-main settings-main">
        <section className="about-hero">
          <BrandMark className="about-mark" />
          <div>
            <h1>
              WeClone
              {version && <span className="about-version">v{version}</span>}
            </h1>
            <p>{t(($) => $.about.tagline)}</p>
            <p className="summary-hint">{t(($) => $.about.local)}</p>
          </div>
        </section>

        <h2 className="about-heading">{t(($) => $.about.project)}</h2>
        <div className="about-grid">
          <LinkCard href={LINKS.home} icon={Globe} title={t(($) => $.about.home)} text="weclone.love" />
          <LinkCard
            href={zh ? LINKS.docs.zh : LINKS.docs.en}
            icon={BookOpen}
            title={t(($) => $.about.docs)}
            text="docs.weclone.love"
          />
          <LinkCard href={LINKS.repo} icon={Github} title="GitHub" text="xming521/WeClone" />
          <LinkCard
            href={LINKS.issues}
            icon={Bug}
            title={t(($) => $.about.issues)}
            text={t(($) => $.about.issuesText)}
          />
        </div>

        <h2 className="about-heading">{t(($) => $.about.community)}</h2>
        <div className="about-grid">
          <LinkCard href={LINKS.qq} icon={Users} title={t(($) => $.about.qq)} text="708067078" />
          <LinkCard href={LINKS.telegram} icon={Send} title="Telegram" text={t(($) => $.about.telegramText)} />
          <LinkCard href={LINKS.x} icon={MessageCircle} title="X (Twitter)" text="@weclone567" />
          <LinkCard
            href={LINKS.xiaohongshu}
            icon={MessageCircle}
            title={t(($) => $.about.xiaohongshu)}
            text="WeClone"
          />
        </div>
        <p className="summary-hint">
          {t(($) => $.about.help)}{' '}
          <a className="wc-link" href={zh ? LINKS.faq.zh : LINKS.faq.en} target="_blank" rel="noreferrer">
            {t(($) => $.about.faq)}
          </a>
        </p>

        <h2 className="about-heading">{t(($) => $.about.credits)}</h2>
        <p className="summary-hint about-credits">
          <Trans
            i18nKey={($) => $.about.licenseLine}
            components={{ license: <a className="wc-link" href={LINKS.license} target="_blank" rel="noreferrer" /> }}
          />
          {THANKS.map(([name, href], index) => (
            <span key={name}>
              {index > 0 && t(($) => $.common.listSeparator)}
              <a className="wc-link" href={href} target="_blank" rel="noreferrer">
                {name}
              </a>
            </span>
          ))}
        </p>
      </main>
    </div>
  );
}

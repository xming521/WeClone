import { Trans } from 'react-i18next';
import i18n from '../i18n';
import type Resources from '../i18n/resources';
import type { Problem } from './progress';

type ProblemCode = keyof Resources['translation']['build']['problems'];

// Problems whose fix the docs site writes up, linked from the <a> of their text.
const DOCS: Partial<Record<ProblemCode, 'telegram' | 'styleDeps'>> = {
  telegram_id_missing: 'telegram',
  style_deps_missing: 'styleDeps',
};

/** A check problem in the page's language when it knows the code, else the server's text. */
export function ProblemText({ problem }: { problem: Problem }) {
  if (!i18n.exists(`build.problems.${problem.code}`)) return problem.message;
  const code = problem.code as ProblemCode;
  const params = Object.fromEntries(
    Object.entries(problem.params).map(([name, value]) => [
      name,
      Array.isArray(value) ? value.join(i18n.t(($) => $.common.listSeparator)) : value,
    ]),
  );
  const page = DOCS[code];
  return (
    <Trans
      i18nKey={($) => $.build.problems[code]}
      values={params}
      components={{
        a: <a className="wc-link" href={page && i18n.t(($) => $.docs[page])} target="_blank" rel="noreferrer" />,
      }}
    />
  );
}

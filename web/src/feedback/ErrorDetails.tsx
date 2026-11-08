import { useTranslation } from 'react-i18next';
import { CopyField } from '../assistant/accessParts';
import '../assistant/assistant.css';

/** What went wrong in the program's own words, folded away and ready to copy into a bug report. */
export function ErrorDetails({ text }: { text: string }) {
  const { t } = useTranslation();
  return (
    <details className="error-details">
      <summary>{t(($) => $.common.errorDetails)}</summary>
      <CopyField label={t(($) => $.common.errorDetails)} text={text} />
    </details>
  );
}

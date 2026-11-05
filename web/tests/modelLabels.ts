import type { ModelLabels } from '../src/data.ts';
import zhCN from '../src/i18n/locales/zh-CN/translation.json' with { type: 'json' };

/** The zh-CN labels the store passes, read straight from the catalog since tests run without i18next. */
export const labels: ModelLabels = {
  root: zhCN.model.root,
  dimensions: zhCN.model.dimensions,
  unassignedDomain: zhCN.model.unassignedDomain,
  directTopic: zhCN.model.directTopic,
};

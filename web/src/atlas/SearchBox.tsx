import { useMemo } from 'react';
import { ArrowUpRight, Search, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { ancestors, dimensionColor, displayProfileText } from '../data';
import { useProfileStore } from '../store';

export function SearchBox() {
  const { t } = useTranslation();
  const viewModel = useProfileStore((state) => state.viewModel)!;
  const search = useProfileStore((state) => state.search);
  const setSearch = useProfileStore((state) => state.setSearch);
  const reveal = useProfileStore((state) => state.reveal);
  const normalized = search.trim().toLocaleLowerCase();
  const results = useMemo(
    () =>
      normalized
        ? [...viewModel.nodes.values()]
            .filter(
              (node) =>
                node.kind !== 'root' &&
                (node.name.toLocaleLowerCase().includes(normalized) ||
                  node.factIndices.some((index) => {
                    const value = viewModel.facts[index].value;
                    return (
                      value.toLocaleLowerCase().includes(normalized) ||
                      displayProfileText(value).toLocaleLowerCase().includes(normalized)
                    );
                  })),
            )
            .slice(0, 8)
        : [],
    [viewModel, normalized],
  );

  return (
    <div className="search-wrap">
      <Search size={18} strokeWidth={1.8} />
      <input
        value={search}
        onChange={(event) => setSearch(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter' && results[0]) reveal(results[0].id);
          if (event.key === 'Escape') setSearch('');
        }}
        placeholder={t(($) => $.atlas.search.placeholder)}
        aria-label={t(($) => $.atlas.search.placeholder)}
      />
      {search && (
        <button
          className="search-clear"
          type="button"
          aria-label={t(($) => $.atlas.search.clear)}
          onClick={() => setSearch('')}
        >
          <X size={16} />
        </button>
      )}
      {normalized && (
        <div className="search-results" role="listbox">
          {results.length ? (
            results.map((node) => (
              <button key={node.id} type="button" role="option" aria-selected="false" onClick={() => reveal(node.id)}>
                <span className="search-result-dot" style={{ backgroundColor: dimensionColor(node.dim) }} />
                <span className="search-result-main">
                  <strong>{node.name}</strong>
                  <small>
                    {ancestors(viewModel, node.id)
                      .slice(1, -1)
                      .map((part) => part.name)
                      .join(' / ')}
                  </small>
                </span>
                <ArrowUpRight size={15} />
              </button>
            ))
          ) : (
            <div className="search-empty">{t(($) => $.atlas.search.empty)}</div>
          )}
        </div>
      )}
    </div>
  );
}

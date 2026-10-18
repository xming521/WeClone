import { useMemo } from 'react';
import { ArrowUpRight, Search, X } from 'lucide-react';
import { ancestors, dimensionColor, displayProfileText } from '../data';
import { useProfileStore } from '../store';

export function SearchBox() {
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
        placeholder="搜索主题、属性或内容"
        aria-label="搜索主题、属性或内容"
      />
      {search && (
        <button className="search-clear" type="button" aria-label="清除搜索" onClick={() => setSearch('')}>
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
            <div className="search-empty">没有找到相关画像</div>
          )}
        </div>
      )}
    </div>
  );
}

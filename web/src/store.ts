import { create } from 'zustand';
import type { ProfileModel } from './data';
import { ancestors } from './data';

interface GraphState {
  focusedDimension: number | null;
  expandedIds: Set<string>;
  selectedId: string | null;
  detailOpen: boolean;
  search: string;
  resetVersion: number;
  setSearch: (value: string) => void;
  initialize: (model: ProfileModel) => void;
  reset: (model: ProfileModel) => void;
  closeDetail: () => void;
  clearSelection: () => void;
  toggleLevel: (model: ProfileModel, level: 1 | 2 | 3) => void;
  toggleExpanded: (model: ProfileModel, id: string) => void;
  reveal: (model: ProfileModel, id: string) => void;
}

function defaultExpandedNodes(model: ProfileModel): Set<string> {
  return new Set(['root', ...model.dimensionIds]);
}

const levelKinds = ['dimension', 'topic', 'subtopic'] as const;

export function levelIsExpanded(model: ProfileModel, expandedIds: Set<string>, level: 1 | 2 | 3, dim: number | null): boolean {
  return dim !== null && expandedIds.has('root') && [...model.nodes.values()]
    .filter((node) => node.dim === dim && node.childIds.length && levelKinds.slice(0, level).some((kind) => node.kind === kind))
    .every((node) => expandedIds.has(node.id));
}

export const useGraphStore = create<GraphState>((set) => ({
  focusedDimension: null,
  expandedIds: new Set(),
  selectedId: 'root',
  detailOpen: false,
  search: '',
  resetVersion: 0,
  setSearch: (search) => set({ search }),
  initialize: (model) => set({ expandedIds: defaultExpandedNodes(model), detailOpen: false }),
  reset: (model) => set((state) => ({
    focusedDimension: null,
    expandedIds: defaultExpandedNodes(model),
    selectedId: 'root',
    detailOpen: false,
    search: '',
    resetVersion: state.resetVersion + 1,
  })),
  closeDetail: () => set({ detailOpen: false }),
  clearSelection: () => set({ selectedId: null, focusedDimension: null, detailOpen: false }),
  toggleLevel: (model, level) => set((state) => {
    const dim = state.focusedDimension;
    if (dim === null) return state;
    const expandedIds = new Set(state.expandedIds);
    if (levelIsExpanded(model, expandedIds, level, dim)) {
      for (const node of model.nodes.values()) {
        if (node.dim !== dim) continue;
        if (node.kind === 'dimension' && level <= 1) expandedIds.delete(node.id);
        if (node.kind === 'topic' && level <= 2) expandedIds.delete(node.id);
        if (node.kind === 'subtopic' && level <= 3) expandedIds.delete(node.id);
      }
    } else {
      expandedIds.add('root');
      for (const node of model.nodes.values()) {
        if (node.dim === dim && node.childIds.length && levelKinds.slice(0, level).some((kind) => node.kind === kind)) expandedIds.add(node.id);
      }
    }
    return { expandedIds };
  }),
  toggleExpanded: (model, id) => set((state) => {
    const node = model.nodes.get(id);
    if (!node) return state;
    if (!node.childIds.length) return { selectedId: id, focusedDimension: node.dim, detailOpen: true };
    const expandedIds = new Set(state.expandedIds);
    if (expandedIds.has(id)) expandedIds.delete(id);
    else expandedIds.add(id);
    return { expandedIds, selectedId: id, focusedDimension: node.dim, detailOpen: true };
  }),
  reveal: (model, id) => set((state) => {
    const node = model.nodes.get(id);
    if (!node) return state;
    const expandedIds = new Set(state.expandedIds);
    for (const item of ancestors(model, id)) {
      if (item.childIds.length) expandedIds.add(item.id);
    }
    return { selectedId: id, focusedDimension: node.dim, expandedIds, search: '', detailOpen: true };
  }),
}));

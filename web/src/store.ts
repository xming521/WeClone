import { create } from 'zustand';
import { api } from './api';
import { ancestors, buildModel, filterProfile, reconcileNavigation, toggleLevelExpansion } from './data';
import type { ChatSelection, ProfileFact, ProfileInput, ProfileModel, ReviewStatus, ReviewView } from './data';

/** What the edit dialog works on. The draft itself is local state of the dialog. */
export interface EditTarget {
  fact: ProfileFact | null;
  /** Default location of a new fact, derived from the current selection. */
  groupId: string;
}

export interface FactDraft {
  value: string;
  attr: string;
  group_id: string;
}

interface ProfileValues {
  // Data
  input: ProfileInput | null;
  view: ReviewView;
  /** Model of the facts matching `view`. */
  viewModel: ProfileModel | null;
  /** Model of all facts regardless of `view`. */
  fullModel: ProfileModel | null;
  error: string;
  busy: boolean;
  // Review UI
  editing: EditTarget | null;
  chat: ChatSelection | null;
  // Navigation
  focusedDimension: number | null;
  expandedIds: Set<string>;
  selectedId: string | null;
  detailOpen: boolean;
  detailTab: 'overview' | 'review';
  search: string;
  resetVersion: number;
}

interface ProfileActions {
  // Data
  load: () => Promise<void>;
  reload: () => Promise<void>;
  refresh: () => Promise<ProfileInput>;
  setView: (view: ReviewView) => void;
  setError: (error: string) => void;
  mutate: (request: () => Promise<unknown>, closeEditor?: boolean) => Promise<boolean>;
  review: (facts: ProfileFact[], status: ReviewStatus) => Promise<boolean>;
  resetSession: () => void;
  // Review UI
  edit: (fact: ProfileFact | null) => void;
  closeEditor: () => void;
  save: (draft: FactDraft, approve: boolean) => Promise<boolean> | undefined;
  openChat: (selection: ChatSelection, opener: HTMLButtonElement) => void;
  closeChat: () => void;
  // Navigation
  setDetailTab: (tab: 'overview' | 'review') => void;
  setSearch: (value: string) => void;
  reset: () => void;
  closeDetail: () => void;
  clearSelection: () => void;
  toggleLevel: (level: 1 | 2 | 3) => void;
  toggleExpanded: (id: string) => void;
  reveal: (id: string) => void;
}

type ProfileState = ProfileValues & ProfileActions;

function defaultExpandedNodes(model: ProfileModel): Set<string> {
  return new Set(['root', ...model.dimensionIds]);
}

function initialValues(): ProfileValues {
  return {
    input: null,
    view: 'all',
    viewModel: null,
    fullModel: null,
    error: '',
    busy: false,
    editing: null,
    chat: null,
    focusedDimension: null,
    expandedIds: new Set(),
    selectedId: 'root',
    detailOpen: false,
    detailTab: 'overview',
    search: '',
    resetVersion: 0,
  };
}

function initialNavigation(viewModel: ProfileModel) {
  return { expandedIds: defaultExpandedNodes(viewModel), detailOpen: false, detailTab: 'overview' as const };
}

/** Initial map state. Bumping `resetVersion` remounts the graph so the layout starts over. */
function resetState(state: ProfileValues, viewModel: ProfileModel) {
  return {
    ...initialNavigation(viewModel),
    focusedDimension: null,
    selectedId: 'root',
    search: '',
    resetVersion: state.resetVersion + 1,
  };
}

// First model initializes navigation, later models keep the surviving expansion and selection.
function navigationFor(state: ProfileValues, viewModel: ProfileModel) {
  return state.viewModel ? reconcileNavigation(state.viewModel, viewModel, state) : initialNavigation(viewModel);
}

// Bumped on logout so responses of requests started in an earlier session never reach the store.
let sessionEpoch = 0;
let saving = false;
// The DOM node stays outside the reactive state: focus returns to it when the chat panel closes.
let chatOpener: HTMLButtonElement | null = null;

export const useProfileStore = create<ProfileState>((set, get) => ({
  ...initialValues(),

  load: async () => {
    const epoch = sessionEpoch;
    try {
      await get().refresh();
    } catch (reason) {
      if (epoch === sessionEpoch) set({ error: (reason as Error).message });
    }
  },
  reload: async () => {
    const epoch = sessionEpoch;
    try {
      await get().refresh();
      if (epoch === sessionEpoch) set({ error: '' });
    } catch (reason) {
      if (epoch === sessionEpoch) set({ error: (reason as Error).message });
    }
  },
  refresh: async () => {
    const epoch = sessionEpoch;
    const input = await api.get<ProfileInput>('profile');
    if (epoch === sessionEpoch) {
      set((state) => {
        const viewModel = buildModel(filterProfile(input, state.view));
        return { input, fullModel: buildModel(input), viewModel, ...navigationFor(state, viewModel) };
      });
    }
    return input;
  },
  setView: (view) =>
    set((state) => {
      if (state.view === view) return state;
      if (!state.input) return { view };
      const viewModel = buildModel(filterProfile(state.input, view));
      return { view, viewModel, ...navigationFor(state, viewModel) };
    }),
  setError: (error) => set({ error }),
  mutate: async (request, closeEditor = false) => {
    if (saving) return false;
    saving = true;
    const epoch = sessionEpoch;
    set({ busy: true, error: '' });
    try {
      await request();
      if (closeEditor && epoch === sessionEpoch) set({ editing: null });
      try {
        await get().refresh();
      } catch {
        if (epoch === sessionEpoch) set({ error: '已保存，但刷新数据失败。请点击“刷新数据”后继续。' });
      }
      return true;
    } catch (reason) {
      if (epoch === sessionEpoch) set({ error: (reason as Error).message });
      return false;
    } finally {
      if (epoch === sessionEpoch) {
        saving = false;
        set({ busy: false });
      }
    }
  },
  review: (facts, status) =>
    get().mutate(() => api.post('reviews/batch', { items: facts.map(({ id, version }) => ({ id, version })), status })),
  resetSession: () => {
    sessionEpoch += 1;
    saving = false;
    chatOpener = null;
    set(initialValues());
  },

  edit: (fact) => {
    const { input, viewModel, selectedId } = get();
    if (!input) return;
    const selected = viewModel?.nodes.get(selectedId ?? 'root');
    const location =
      input.locations.find((item) => item.id === selected?.id) ??
      input.locations.find((item) => item.id === selected?.parentId) ??
      input.locations[0];
    set({
      error: '',
      editing: { fact, groupId: location?.id ?? '' },
    });
  },
  closeEditor: () => set({ editing: null, error: '' }),
  save: ({ value, attr, group_id }, approve) => {
    const { editing, mutate } = get();
    if (!editing) return;
    const { fact } = editing;
    const body = {
      value,
      attr,
      group_id,
      approve,
      ...(fact ? { version: fact.version } : {}),
    };
    return mutate(() => (fact ? api.patch(`facts/${fact.id}`, body) : api.post('facts', body)), true);
  },
  openChat: (chat, opener) => {
    chatOpener = opener;
    set({ chat });
  },
  closeChat: () => {
    set({ chat: null });
    if (chatOpener?.isConnected) chatOpener.focus();
  },

  setDetailTab: (detailTab) => set({ detailTab }),
  setSearch: (search) => set({ search }),
  reset: () => set((state) => (state.viewModel ? resetState(state, state.viewModel) : state)),
  closeDetail: () => set({ detailOpen: false }),
  clearSelection: () => set({ selectedId: null, focusedDimension: null, detailOpen: false }),
  toggleLevel: (level) =>
    set((state) => {
      const model = state.viewModel;
      const dim = state.focusedDimension;
      if (!model || dim === null) return state;
      return { expandedIds: toggleLevelExpansion(model, state.expandedIds, level, dim) };
    }),
  toggleExpanded: (id) =>
    set((state) => {
      const node = state.viewModel?.nodes.get(id);
      if (!node) return state;
      const detailTab = node.kind === 'attribute' ? 'review' : 'overview';
      if (!node.childIds.length) return { selectedId: id, focusedDimension: node.dim, detailOpen: true, detailTab };
      // Reopening the collapsed root redraws the initial map, since re-expanding from the collapsed state scrambles it.
      if (id === 'root' && !state.expandedIds.has(id))
        return { ...resetState(state, state.viewModel!), detailOpen: true, detailTab };
      const expandedIds = new Set(state.expandedIds);
      if (expandedIds.has(id)) expandedIds.delete(id);
      else expandedIds.add(id);
      return { expandedIds, selectedId: id, focusedDimension: node.dim, detailOpen: true, detailTab };
    }),
  reveal: (id) =>
    set((state) => {
      const model = state.viewModel;
      const node = model?.nodes.get(id);
      if (!model || !node) return state;
      const expandedIds = new Set(state.expandedIds);
      for (const item of ancestors(model, id)) {
        if (item.childIds.length) expandedIds.add(item.id);
      }
      return {
        selectedId: id,
        focusedDimension: node.dim,
        expandedIds,
        search: '',
        detailOpen: true,
        detailTab: node.kind === 'attribute' ? 'review' : 'overview',
      };
    }),
}));

// Any change of the selected node, the detail tab or the detail visibility closes the source chat panel.
useProfileStore.subscribe((state, previous) => {
  if (
    state.chat &&
    (state.detailTab !== previous.detailTab ||
      state.selectedId !== previous.selectedId ||
      state.detailOpen !== previous.detailOpen)
  )
    useProfileStore.setState({ chat: null });
});

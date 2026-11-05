import { create } from 'zustand';
import { api, errorCode } from './api';
import {
  DIMENSIONS,
  DOMAIN_ONLY_DIMENSIONS,
  REVIEW_SORTS,
  PRIVACY_LEVELS,
  ancestors,
  buildModel,
  filterProfile,
  reconcileNavigation,
  toggleLevelExpansion,
} from './data';
import type {
  ChatSelection,
  DimensionId,
  Grouping,
  ModelLabels,
  PrivacyLevel,
  ProfileFact,
  ProfileInput,
  ProfileModel,
  ReviewSort,
  ReviewStatus,
  ReviewView,
} from './data';
import { toast, useToasts } from './feedback/toast';
import i18n from './i18n';

/** Names of the nodes the model makes itself, in the current language. */
function modelLabels(): ModelLabels {
  return {
    root: i18n.t(($) => $.model.root),
    dimensions: Object.fromEntries(
      [...DIMENSIONS.map(({ dim }) => dim), ...DOMAIN_ONLY_DIMENSIONS].map((dim) => [
        dim,
        i18n.t(($) => $.model.dimensions[dim]),
      ]),
    ) as Record<DimensionId, string>,
    unassignedDomain: i18n.t(($) => $.model.unassignedDomain),
    directTopic: i18n.t(($) => $.model.directTopic),
  };
}

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
  /** First level of the map: the five dimensions or the person's life domains. */
  grouping: Grouping;
  /** Model of the facts matching `view`. */
  viewModel: ProfileModel | null;
  /** Model of all facts regardless of `view`. */
  fullModel: ProfileModel | null;
  /** Privacy levels the review list shows; none chosen shows every level. Unlike `view`, the map ignores it. */
  privacy: PrivacyLevel[];
  /** Order of the review list, kept across visits. */
  sort: ReviewSort;
  /** The review list marks the card its shortcuts act on, once a shortcut was used or a review round started. */
  reviewKeys: boolean;
  error: string;
  /** The server has no profile yet: none was built, or none is open. */
  missing: boolean;
  /** A write is running or waiting its turn. */
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
  togglePrivacy: (level: PrivacyLevel) => void;
  setSort: (sort: ReviewSort) => void;
  showReviewKeys: () => void;
  /** Opens every pending fact in the review list, the first one marked for the shortcuts. */
  startReview: () => void;
  setGrouping: (grouping: Grouping) => void;
  setError: (error: string) => void;
  mutate: (request: () => Promise<unknown>, closeEditor?: boolean) => Promise<boolean>;
  /** Sets the facts' status. Unless `undoable` is false, a notice offers to put back the statuses they had. */
  review: (facts: ProfileFact[], status: ReviewStatus, undoable?: boolean) => Promise<boolean>;
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
  clickNode: (id: string) => void;
  reveal: (id: string) => void;
}

type ProfileState = ProfileValues & ProfileActions;

function defaultExpandedNodes(model: ProfileModel): Set<string> {
  return new Set(['root', ...model.dimensionIds]);
}

const GROUPING_KEY = 'weclone.profile.grouping';

function savedGrouping(): Grouping {
  try {
    return window.localStorage.getItem(GROUPING_KEY) === 'domain' ? 'domain' : 'dimension';
  } catch {
    return 'dimension';
  }
}

function saveGrouping(grouping: Grouping) {
  try {
    window.localStorage.setItem(GROUPING_KEY, grouping);
  } catch {
    // Without storage the choice lasts for this page only.
  }
}

const PRIVACY_KEY = 'weclone.review.privacy';

/** The privacy levels the review list starts with, chosen in the settings; none shows every level. */
export function savedPrivacy(): PrivacyLevel[] {
  try {
    const saved: unknown = JSON.parse(window.localStorage.getItem(PRIVACY_KEY) ?? '[]');
    return PRIVACY_LEVELS.filter((level) => Array.isArray(saved) && saved.includes(level));
  } catch {
    return [];
  }
}

export function saveDefaultPrivacy(levels: PrivacyLevel[]) {
  try {
    window.localStorage.setItem(PRIVACY_KEY, JSON.stringify(levels));
  } catch {
    // Without storage the choice lasts for this page only.
  }
  useProfileStore.setState({ privacy: levels });
}

const SORT_KEY = 'weclone.review.sort';

function savedSort(): ReviewSort {
  try {
    const saved = window.localStorage.getItem(SORT_KEY);
    return REVIEW_SORTS.find((sort) => sort === saved) ?? 'default';
  } catch {
    return 'default';
  }
}

function saveSort(sort: ReviewSort) {
  try {
    window.localStorage.setItem(SORT_KEY, sort);
  } catch {
    // Without storage the choice lasts for this page only.
  }
}

function initialValues(): ProfileValues {
  return {
    input: null,
    view: 'all',
    grouping: savedGrouping(),
    viewModel: null,
    fullModel: null,
    privacy: savedPrivacy(),
    sort: savedSort(),
    reviewKeys: false,
    error: '',
    missing: false,
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
// Writes run one at a time in the order they were asked for, each after the previous one's refresh.
let writes: Promise<unknown> = Promise.resolve();
let pendingWrites = 0;
// The status each fact with a review still running or waiting will have, laid over every profile
// read until then. `write` tells a later review of the same fact from an earlier one.
const reviewing = new Map<string, { status: ReviewStatus; write: number }>();
let reviewCount = 0;

function withReviews(input: ProfileInput): ProfileInput {
  if (!reviewing.size) return input;
  return {
    ...input,
    facts: input.facts.map((fact) => {
      const pending = reviewing.get(fact.id);
      return pending ? { ...fact, status: pending.status } : fact;
    }),
  };
}

function modelsFor(state: ProfileValues, input: ProfileInput) {
  const viewModel = buildModel(filterProfile(input, state.view), modelLabels(), state.grouping);
  return {
    input,
    fullModel: buildModel(input, modelLabels(), state.grouping),
    viewModel,
    ...navigationFor(state, viewModel),
  };
}
// The DOM node stays outside the reactive state: focus returns to it when the chat panel closes.
let chatOpener: HTMLButtonElement | null = null;

export const useProfileStore = create<ProfileState>((set, get) => ({
  ...initialValues(),

  load: async () => {
    const epoch = sessionEpoch;
    try {
      await get().refresh();
      if (epoch === sessionEpoch) set({ missing: false });
    } catch (reason) {
      if (epoch === sessionEpoch)
        set({ error: (reason as Error).message, missing: errorCode(reason) === 'profile_missing' });
    }
  },
  reload: async () => {
    const epoch = sessionEpoch;
    try {
      await get().refresh();
      if (epoch === sessionEpoch) set({ error: '', missing: false });
    } catch (reason) {
      if (epoch === sessionEpoch)
        set({ error: (reason as Error).message, missing: errorCode(reason) === 'profile_missing' });
    }
  },
  refresh: async () => {
    const epoch = sessionEpoch;
    const input = withReviews(await api.get<ProfileInput>('profile'));
    if (epoch === sessionEpoch) set((state) => modelsFor(state, input));
    return input;
  },
  setView: (view) =>
    set((state) => {
      if (state.view === view) return state;
      if (!state.input) return { view };
      const viewModel = buildModel(filterProfile(state.input, view), modelLabels(), state.grouping);
      return { view, viewModel, ...navigationFor(state, viewModel) };
    }),
  togglePrivacy: (level) =>
    set((state) => ({
      privacy: state.privacy.includes(level)
        ? state.privacy.filter((item) => item !== level)
        : [...state.privacy, level],
    })),
  setSort: (sort) => {
    saveSort(sort);
    set({ sort });
  },
  showReviewKeys: () => set((state) => (state.reviewKeys ? state : { reviewKeys: true })),
  startReview: () => {
    get().setView('pending');
    set({ selectedId: 'root', focusedDimension: null, detailOpen: true, detailTab: 'review', reviewKeys: true });
  },
  // The other grouping is a different map: it starts over from the overview.
  setGrouping: (grouping) =>
    set((state) => {
      if (state.grouping === grouping) return state;
      saveGrouping(grouping);
      if (!state.input) return { grouping };
      const viewModel = buildModel(filterProfile(state.input, state.view), modelLabels(), grouping);
      return {
        grouping,
        viewModel,
        fullModel: buildModel(state.input, modelLabels(), grouping),
        ...resetState(state, viewModel),
      };
    }),
  setError: (error) => set({ error }),
  mutate: (request, closeEditor = false) => {
    const epoch = sessionEpoch;
    pendingWrites += 1;
    set({ busy: true });
    const run = async () => {
      if (epoch !== sessionEpoch) return false;
      set({ error: '' });
      try {
        await request();
        if (closeEditor && epoch === sessionEpoch) set({ editing: null });
        try {
          await get().refresh();
        } catch {
          if (epoch === sessionEpoch) set({ error: i18n.t(($) => $.review.refreshFailed) });
        }
        return true;
      } catch (reason) {
        if (epoch === sessionEpoch) set({ error: (reason as Error).message });
        return false;
      } finally {
        if (epoch === sessionEpoch && !--pendingWrites) set({ busy: false });
      }
    };
    const result = writes.then(run);
    writes = result;
    return result;
  },
  // The list and the map take the new status on the click; the write follows in its turn. If it
  // fails, the refresh after it brings back the stored status.
  review: async (facts, status, undoable = true) => {
    const asked = new Set(facts.map((fact) => fact.id));
    // The statuses before this review, as the page shows them: earlier reviews still running count. Facts that
    // already have the status are left out, and with none left there is nothing to write.
    const before = new Map<ReviewStatus, ProfileFact[]>();
    const ids = new Set<string>();
    for (const fact of get().input?.facts ?? []) {
      if (!asked.has(fact.id) || fact.status === status) continue;
      before.set(fact.status, [...(before.get(fact.status) ?? []), fact]);
      ids.add(fact.id);
    }
    if (!ids.size) return true;
    const write = ++reviewCount;
    for (const id of ids) reviewing.set(id, { status, write });
    const { input } = get();
    if (input) set((state) => modelsFor(state, withReviews(input)));
    const settle = () => {
      for (const id of ids) if (reviewing.get(id)?.write === write) reviewing.delete(id);
    };
    const notice = undoable
      ? toast(
          i18n.t(($) => $.toast[status], { count: ids.size }),
          {
            undo: before.size
              ? () => {
                  for (const [previous, group] of before) void get().review(group, previous, false);
                  toast(i18n.t(($) => $.toast.undone));
                }
              : undefined,
          },
        )
      : null;
    const ok = await get().mutate(() => {
      // Read the versions when the write runs: the writes before it have changed them.
      const items = get()
        .input!.facts.filter((fact) => ids.has(fact.id))
        .map(({ id, version }) => ({ id, version }));
      return api.post('reviews/batch', { items, status }).finally(settle);
    });
    if (!ok) {
      // The error shows instead; there is nothing to take back.
      if (notice !== null) useToasts.getState().dismiss(notice);
      await get()
        .refresh()
        .catch(() => undefined);
    }
    return ok;
  },
  resetSession: () => {
    sessionEpoch += 1;
    writes = Promise.resolve();
    pendingWrites = 0;
    reviewing.clear();
    useToasts.getState().dismiss();
    chatOpener = null;
    set(initialValues());
  },

  edit: (fact) => {
    const { input, viewModel, selectedId } = get();
    if (!input) return;
    const selected = viewModel?.nodes.get(selectedId ?? 'root');
    // An attribute's facts name its stored group; in the domain view its parent is a merged topic.
    const parentId =
      selected?.kind === 'attribute' ? viewModel!.facts[selected.factIndices[0]]?.group_id : selected?.parentId;
    const location =
      input.locations.find((item) => item.id === selected?.id) ??
      input.locations.find((item) => item.id === parentId) ??
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
    const body = { value, attr, group_id, approve };
    const saved = (ok: boolean) => {
      if (ok) toast(fact ? i18n.t(($) => $.toast.saved) : i18n.t(($) => $.toast.added));
      return ok;
    };
    if (!fact) return mutate(() => api.post('facts', body), true).then(saved);
    // Read the version when the write runs: a review clicked before the editor opened may still be running.
    const version = () => get().input?.facts.find((item) => item.id === fact.id)?.version ?? fact.version;
    return mutate(() => api.patch(`facts/${fact.id}`, { ...body, version: version() }), true).then(saved);
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
  // A click on the map selects the node and expands it. Only a click on the node already selected
  // collapses it, so looking at a node never hides its branch. The root always stays expanded.
  clickNode: (id) =>
    set((state) => {
      const node = state.viewModel?.nodes.get(id);
      if (!node) return state;
      const selection = {
        selectedId: id,
        focusedDimension: node.dim,
        detailOpen: true,
        detailTab: node.kind === 'attribute' ? ('review' as const) : ('overview' as const),
      };
      const expanded = state.expandedIds.has(id);
      if (!node.childIds.length || (expanded && (id === 'root' || id !== state.selectedId))) return selection;
      const expandedIds = new Set(state.expandedIds);
      if (expanded) expandedIds.delete(id);
      else expandedIds.add(id);
      return { expandedIds, ...selection };
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

// The model names the nodes it makes itself, so another language rebuilds it. Node ids stay the same, and the map
// keeps what is expanded and selected.
i18n.on('languageChanged', () =>
  useProfileStore.setState((state) => {
    if (!state.input) return state;
    const viewModel = buildModel(filterProfile(state.input, state.view), modelLabels(), state.grouping);
    return {
      viewModel,
      fullModel: buildModel(state.input, modelLabels(), state.grouping),
      ...navigationFor(state, viewModel),
    };
  }),
);

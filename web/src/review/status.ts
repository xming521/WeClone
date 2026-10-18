import { BadgeCheck, CircleX, FileClock, Layers3 } from 'lucide-react';
import type { ReviewView } from '../data';

export const STATUS: Record<ReviewView, string> = {
  all: '全部',
  pending: '待审核',
  approved: '已采纳',
  rejected: '未采纳',
};
export const VIEW_ICONS = { all: Layers3, pending: FileClock, approved: BadgeCheck, rejected: CircleX };

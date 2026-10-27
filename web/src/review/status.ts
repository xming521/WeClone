import { BadgeCheck, CircleX, FileClock, Layers3 } from 'lucide-react';
import type { ReviewView } from '../data';

export const REVIEW_VIEWS: ReviewView[] = ['all', 'pending', 'approved', 'rejected'];
export const VIEW_ICONS = { all: Layers3, pending: FileClock, approved: BadgeCheck, rejected: CircleX };

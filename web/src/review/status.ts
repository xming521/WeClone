import type { ReviewStatus, ReviewView } from '../data';

export const REVIEW_VIEWS: ReviewView[] = ['all', 'pending', 'approved', 'rejected'];

/** The design system names an approved fact `adopted`: the `data-status` value of its marks. */
export const MARK_STATUS: Record<ReviewStatus, 'pending' | 'adopted' | 'rejected'> = {
  pending: 'pending',
  approved: 'adopted',
  rejected: 'rejected',
};

import { apiRequest, ServiceLogger } from './api';
import type {
  ProductReview,
  ReviewSummary,
  ReviewSummarization,
} from '../types';

export const listProductReviews = async (
  productId: string,
  limit: number = 100,
  offset: number = 0
): Promise<ProductReview[]> => {
  ServiceLogger.logServiceCall('listProductReviews', {
    productId,
    limit,
    offset,
  });
  return apiRequest<ProductReview[]>(
    `/products/${encodeURIComponent(productId)}/reviews?limit=${limit}&offset=${offset}`,
    'listProductReviews'
  );
};

export const getProductReviewSummary = async (
  productId: string
): Promise<ReviewSummary> => {
  ServiceLogger.logServiceCall('getProductReviewSummary', { productId });
  return apiRequest<ReviewSummary>(
    `/products/${encodeURIComponent(productId)}/reviews/summary`,
    'getProductReviewSummary'
  );
};

export const summarizeProductReviews = async (
  productId: string
): Promise<ReviewSummarization> => {
  ServiceLogger.logServiceCall('summarizeProductReviews', { productId });
  return apiRequest<ReviewSummarization>(
    `/products/${encodeURIComponent(productId)}/reviews/summarize`,
    'summarizeProductReviews'
  );
};

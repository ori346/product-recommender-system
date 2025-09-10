import {
  PageSection,
  Title,
  Spinner,
  EmptyState,
  EmptyStateBody,
} from '@patternfly/react-core';
import { LazyProductGallery } from './LazyProductGallery';
import type { ProductData } from '../types';

interface ImageSearchResultsProps {
  products: ProductData[];
  isLoading: boolean;
  error: any;
}

export function ImageSearchResults({
  products,
  isLoading,
  error,
}: ImageSearchResultsProps) {
  if (isLoading) {
    return (
      <PageSection hasBodyWrapper={false}>
        <Title headingLevel={'h1'} style={{ marginTop: '15px' }}>
          Similar Products Found
        </Title>
        <div style={{ textAlign: 'center', marginTop: '32px' }}>
          <Spinner size='lg' />
          <div style={{ marginTop: '16px', color: '#666' }}>
            Loading similar products...
          </div>
        </div>
      </PageSection>
    );
  }

  if (error) {
    return (
      <PageSection hasBodyWrapper={false}>
        <Title headingLevel={'h1'} style={{ marginTop: '15px' }}>
          Similar Products Found
        </Title>
        <EmptyState>
          <Title headingLevel='h4' size='lg'>
            Error in Image Search
          </Title>
          <EmptyStateBody>
            There was an error while searching. Please try again.
            {error instanceof Error && (
              <div
                style={{
                  marginTop: '8px',
                  fontStyle: 'italic',
                  fontSize: '14px',
                  opacity: 0.8,
                }}
              >
                {error.message}
              </div>
            )}
          </EmptyStateBody>
        </EmptyState>
      </PageSection>
    );
  }

  return (
    <>
      <PageSection hasBodyWrapper={false}>
        <Title headingLevel={'h1'} style={{ marginTop: '15px' }}>
          Similar Products Found
        </Title>

        {products.length === 0 ? (
          <EmptyState>
            <Title headingLevel='h4' size='lg'>
              No similar products found
            </Title>
            <EmptyStateBody>
              No similar products found. Try a different image or search
              criteria.
            </EmptyStateBody>
          </EmptyState>
        ) : (
          <LazyProductGallery
            products={products}
            showProductCount={true}
            showScrollToTop={true}
          />
        )}
      </PageSection>
    </>
  );
}

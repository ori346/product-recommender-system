import { useState } from 'react';
import {
  Button,
  InputGroup,
  TextInput,
  Tooltip,
  ToggleGroup,
  ToggleGroupItem,
  FileUpload,
} from '@patternfly/react-core';
import { LinkIcon, UploadIcon, ImageIcon } from '@patternfly/react-icons';
import {
  useProductSearchByImageLink,
  useProductSearchByImage,
} from '../hooks/useProducts';
import { ImageSearchResults } from './image-search-results';

export const ImageSearch: React.FC = () => {
  const [searchType, setSearchType] = useState<'url' | 'file'>('url');
  const [imageUrl, setImageUrl] = useState('');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [filename, setFilename] = useState('');
  const [urlSearchTrigger, setUrlSearchTrigger] = useState('');
  const [fileSearchTrigger, setFileSearchTrigger] = useState<File | null>(null);
  const [isSearching, setIsSearching] = useState(false);

  const {
    data: urlData,
    error: urlError,
    isLoading: urlLoading,
  } = useProductSearchByImageLink(urlSearchTrigger, 10, !!urlSearchTrigger);
  const {
    data: fileData,
    error: fileError,
    isLoading: fileLoading,
  } = useProductSearchByImage(fileSearchTrigger, 10, !!fileSearchTrigger);

  // Use the appropriate data/error/loading based on search type
  const data = searchType === 'url' ? urlData : fileData;
  const error = searchType === 'url' ? urlError : fileError;
  const isLoading = searchType === 'url' ? urlLoading : fileLoading;

  const handleUrlSearch = async () => {
    if (!imageUrl.trim()) return;
    setIsSearching(true);
    setFileSearchTrigger(null); // Clear file search
    setUrlSearchTrigger(imageUrl);
    setIsSearching(false);
  };

  const handleFileSearch = async () => {
    if (!imageFile) return;
    setIsSearching(true);
    setUrlSearchTrigger(''); // Clear URL search
    setFileSearchTrigger(imageFile);
    setIsSearching(false);
  };

  const handleFileChange = (_event: any, file: File) => {
    setImageFile(file);
    setFilename(file.name);
  };

  const handleFileClear = () => {
    setImageFile(null);
    setFilename('');
    setFileSearchTrigger(null);
  };

  return (
    <div>
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          marginBottom: '24px',
        }}
      >
        <ToggleGroup
          aria-label='Search type selection'
          style={{
            background: 'white',
            borderRadius: '12px',
            padding: '4px',
            boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
            border: '1px solid #e9ecef',
          }}
        >
          <ToggleGroupItem
            icon={<LinkIcon />}
            text='Image URL'
            isSelected={searchType === 'url'}
            onChange={() => setSearchType('url')}
            style={{
              borderRadius: '8px',
              padding: '12px 20px',
              fontWeight: '500',
              transition: 'all 0.2s ease',
              border: 'none',
              ...(searchType === 'url' && {
                background: 'linear-gradient(135deg, #3498db 0%, #2980b9 100%)',
                color: 'white',
                boxShadow: '0 2px 4px rgba(52, 152, 219, 0.3)',
              }),
            }}
          />
          <ToggleGroupItem
            icon={<ImageIcon />}
            text='Upload Image'
            isSelected={searchType === 'file'}
            onChange={() => setSearchType('file')}
            style={{
              borderRadius: '8px',
              padding: '12px 20px',
              fontWeight: '500',
              transition: 'all 0.2s ease',
              border: 'none',
              ...(searchType === 'file' && {
                background: 'linear-gradient(135deg, #e74c3c 0%, #c0392b 100%)',
                color: 'white',
                boxShadow: '0 2px 4px rgba(231, 76, 60, 0.3)',
              }),
            }}
          />
        </ToggleGroup>
      </div>

      {searchType === 'url' ? (
        <div style={{ marginBottom: '24px' }}>
          <InputGroup style={{ gap: '8px' }}>
            <TextInput
              value={imageUrl}
              onChange={(_event, value) => setImageUrl(value)}
              onKeyDown={event => {
                if (event.key === 'Enter') {
                  handleUrlSearch();
                }
              }}
              placeholder='Enter image URL to find similar products...'
              type='url'
              aria-label='Image URL input'
              style={{
                borderRadius: '8px',
                border: '2px solid #e9ecef',
                padding: '12px 16px',
                fontSize: '16px',
                transition: 'border-color 0.2s ease',
                boxShadow: '0 2px 8px rgba(0,0,0,0.05)',
              }}
            />
            <Tooltip content='Search by image URL'>
              <Button
                variant='primary'
                onClick={handleUrlSearch}
                isDisabled={!imageUrl.trim() || isSearching}
                icon={<LinkIcon />}
                style={{
                  background:
                    'linear-gradient(135deg, #3498db 0%, #2980b9 100%)',
                  border: 'none',
                  borderRadius: '8px',
                  padding: '12px 24px',
                  fontWeight: '600',
                  boxShadow: '0 2px 4px rgba(52, 152, 219, 0.3)',
                  transition: 'all 0.2s ease',
                }}
              >
                Search Similar
              </Button>
            </Tooltip>
          </InputGroup>
        </div>
      ) : (
        <div style={{ marginBottom: '24px' }}>
          <FileUpload
            id='image-file-upload'
            value={imageFile || undefined}
            filename={filename}
            filenamePlaceholder='Drag and drop an image file or upload one'
            onFileInputChange={handleFileChange}
            onClearClick={handleFileClear}
            browseButtonText='Upload Image'
            accept='image/*'
            allowEditingUploadedText={false}
            style={{
              border: '2px dashed #e9ecef',
              borderRadius: '12px',
              padding: '32px',
              textAlign: 'center',
              transition: 'border-color 0.2s ease',
              background: '#fafafa',
            }}
          />
          {imageFile && (
            <div style={{ textAlign: 'center', marginTop: '16px' }}>
              <Button
                variant='primary'
                onClick={handleFileSearch}
                isDisabled={isSearching}
                icon={<UploadIcon />}
                style={{
                  background:
                    'linear-gradient(135deg, #e74c3c 0%, #c0392b 100%)',
                  border: 'none',
                  borderRadius: '8px',
                  padding: '12px 24px',
                  fontWeight: '600',
                  boxShadow: '0 2px 4px rgba(231, 76, 60, 0.3)',
                  transition: 'all 0.2s ease',
                }}
              >
                Search Similar
              </Button>
            </div>
          )}
        </div>
      )}

      {/* Display results or loading state */}
      {(urlSearchTrigger || fileSearchTrigger) && (
        <ImageSearchResults
          products={data || []}
          isLoading={isLoading}
          error={error}
        />
      )}
    </div>
  );
};

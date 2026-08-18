import { describe, it, expect } from 'vitest';
import { normalizeChatResponse, getSourceTypeLabel } from '../utils/responseParser';

describe('responseParser utils', () => {
  it('handles empty or undefined response gracefully', () => {
    const result = normalizeChatResponse(null);
    expect(result.answer).toContain('Received an empty response');
    expect(result.sources).toEqual([]);
    expect(result.hasSources).toBe(false);
    expect(result.hasWebFallback).toBe(false);
  });

  it('normalizes standard FastAPI chat response', () => {
    const backendData = {
      answer: 'Our company provides AI integrations.',
      sources: [
        {
          title: 'Company Overview',
          source_type: 'pdf',
          url: 'https://example.com/overview.pdf',
          page: 2
        }
      ]
    };

    const result = normalizeChatResponse(backendData);
    expect(result.answer).toBe('Our company provides AI integrations.');
    expect(result.hasSources).toBe(true);
    expect(result.sources).toHaveLength(1);
    expect(result.sources[0].title).toBe('Company Overview');
    expect(result.sources[0].url).toBe('https://example.com/overview.pdf');
    expect(result.sources[0].page).toBe(2);
    expect(result.sources[0].sourceType).toBe('pdf');
    expect(result.hasWebFallback).toBe(false);
  });

  it('detects web fallback in sources or payload', () => {
    const backendData = {
      answer: 'Information found on website.',
      sources: [
        {
          title: 'Home Page',
          source_type: 'web',
          url: 'https://example.com'
        }
      ]
    };

    const result = normalizeChatResponse(backendData);
    expect(result.hasWebFallback).toBe(true);
  });

  it('formats badge types correctly', () => {
    expect(getSourceTypeLabel('pdf')).toBe('PDF');
    expect(getSourceTypeLabel('docx')).toBe('DOC');
    expect(getSourceTypeLabel('markdown')).toBe('DOCS');
    expect(getSourceTypeLabel('web')).toBe('WEB');
    expect(getSourceTypeLabel('web_fallback')).toBe('WEB');
    expect(getSourceTypeLabel('csv')).toBe('DATA');
    expect(getSourceTypeLabel('other')).toBe('DOC');
  });
});

/**
 * Utilities for normalizing and formatting backend RAG responses for presentation.
 */

/**
 * Normalizes raw API response into a clean UI model.
 * @param {Object} rawData - Response from FastAPI /rag/chat
 * @returns {{
 *   answer: string,
 *   sources: Array<{ id: string, title: string, url: string|null, page: number|null, sourceType: string }>,
 *   hasSources: boolean,
 *   hasWebFallback: boolean
 * }}
 */
export function normalizeChatResponse(rawData) {
  if (!rawData || typeof rawData !== 'object') {
    return {
      answer: 'Received an empty response from the assistant.',
      sources: [],
      hasSources: false,
      hasWebFallback: false,
    };
  }

  const answer = typeof rawData.answer === 'string' && rawData.answer.trim()
    ? rawData.answer.trim()
    : 'No answer was generated for your request.';

  const rawSources = Array.isArray(rawData.sources) ? rawData.sources : [];

  let isWebFallbackUsed = Boolean(rawData.web_fallback_used);

  const sources = rawSources.map((source, index) => {
    const rawType = (source.source_type || source.sourceType || 'document').toLowerCase();
    
    if (rawType.includes('web')) {
      isWebFallbackUsed = true;
    }

    // Sanitize and normalize URL
    let url = null;
    if (typeof source.url === 'string' && source.url.trim()) {
      const trimmedUrl = source.url.trim();
      if (/^https?:\/\//i.test(trimmedUrl)) {
        url = trimmedUrl;
      }
    }

    return {
      id: `source-${index}-${Date.now()}`,
      title: source.title || (url ? formatDomainTitle(url) : `Document Reference ${index + 1}`),
      url: url,
      page: typeof source.page === 'number' ? source.page : null,
      sourceType: rawType,
    };
  });

  return {
    answer,
    sources,
    hasSources: sources.length > 0,
    hasWebFallback: isWebFallbackUsed,
  };
}

/**
 * Extracts a readable domain title from a URL.
 * @param {string} urlString
 * @returns {string}
 */
function formatDomainTitle(urlString) {
  try {
    const parsed = new URL(urlString);
    return `${parsed.hostname}${parsed.pathname.length > 1 ? parsed.pathname : ''}`;
  } catch {
    return urlString;
  }
}

/**
 * Formats a source badge label for clean display.
 * @param {string} sourceType
 * @returns {string}
 */
export function getSourceTypeLabel(sourceType) {
  switch ((sourceType || '').toLowerCase()) {
    case 'pdf':
      return 'PDF';
    case 'docx':
    case 'doc':
      return 'DOC';
    case 'markdown':
    case 'md':
      return 'DOCS';
    case 'web':
    case 'web_fallback':
      return 'WEB';
    case 'csv':
      return 'DATA';
    default:
      return 'DOC';
  }
}

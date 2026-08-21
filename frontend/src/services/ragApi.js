/**
 * API Service for communicating with the FastAPI RAG backend.
 * Handles HTTP requests, timeouts, error normalization, and health checks.
 */

const API_BASE_URL = (import.meta.env.VITE_RAG_API_URL || 'http://localhost:8000').replace(/\/+$/, '');
const DEFAULT_TIMEOUT_MS = 45000; // 45 seconds for retrieval + LLM generation

/**
 * Custom API Error class for normalized frontend error representation.
 */
export class RagApiError extends Error {
  constructor(message, statusCode = null, isNetworkError = false) {
    super(message);
    this.name = 'RagApiError';
    this.statusCode = statusCode;
    this.isNetworkError = isNetworkError;
  }
}

/**
 * Check if the FastAPI service is healthy and reachable.
 * @returns {Promise<{ online: boolean, service?: string }>}
 */
export async function checkHealth() {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 6000);

  try {
    const response = await fetch(`${API_BASE_URL}/rith`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (response.ok) {
      const data = await response.json();
      return { online: true, service: data.service || 'rag-fastapi' };
    }
    return { online: false };
  } catch (err) {
    clearTimeout(timeoutId);
    return { online: false, error: err.message };
  }
}

/**
 * Send a chat message to the FastAPI RAG backend.
 * @param {string} message - User's question
 * @param {Object} [options]
 * @param {AbortSignal} [options.signal] - Optional external abort signal
 * @returns {Promise<{ answer: string, sources: Array }>}
 */
export async function sendChatMessage(message, options = {}) {
  const trimmed = (message || '').trim();
  if (!trimmed) {
    throw new RagApiError('Please enter a question before sending.', 400);
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT_MS);

  // Link external signal if provided
  if (options.signal) {
    options.signal.addEventListener('abort', () => controller.abort());
  }

  try {
    const response = await fetch(`${API_BASE_URL}/rag/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify({ message: trimmed }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      let errorMessage = 'An unexpected error occurred.';
      try {
        const errorData = await response.json();
        if (errorData?.detail) {
          if (Array.isArray(errorData.detail)) {
            // FastAPI Pydantic validation error format
            errorMessage = errorData.detail.map(d => d.msg || d.message).join(', ');
          } else if (typeof errorData.detail === 'string') {
            errorMessage = errorData.detail;
          }
        }
      } catch {
        // Fallback for non-JSON error bodies
        if (response.status === 404) {
          errorMessage = 'The chat endpoint was not found on the server.';
        } else if (response.status === 500) {
          errorMessage = 'The chatbot backend encountered a processing error. Please try again.';
        } else if (response.status === 401) {
          errorMessage = 'Service authorization failed.';
        }
      }

      throw new RagApiError(errorMessage, response.status);
    }

    const data = await response.json();
    return data;
  } catch (err) {
    clearTimeout(timeoutId);

    if (err instanceof RagApiError) {
      throw err;
    }

    if (err.name === 'AbortError') {
      throw new RagApiError('The request timed out while waiting for a response. Please try again.', 408, true);
    }

    // Network error (backend offline, CORS failure, or connection refused)
    throw new RagApiError(
      'Unable to connect to the chatbot service. Please ensure the backend is running.',
      null,
      true
    );
  }
}

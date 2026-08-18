import { useState, useCallback, useEffect, useRef } from 'react';
import { sendChatMessage, checkHealth } from '../services/ragApi';
import { normalizeChatResponse } from '../utils/responseParser';

/**
 * Custom React hook for managing chatbot state, API interactions, and error handling.
 */
export function useChat() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastFailedMessage, setLastFailedMessage] = useState(null);
  const [backendStatus, setBackendStatus] = useState('checking'); // 'online' | 'offline' | 'checking'

  const isMounted = useRef(true);

  // Ping backend on mount to check connectivity
  const refreshBackendStatus = useCallback(async () => {
    try {
      const result = await checkHealth();
      if (isMounted.current) {
        setBackendStatus(result.online ? 'online' : 'offline');
      }
    } catch {
      if (isMounted.current) {
        setBackendStatus('offline');
      }
    }
  }, []);

  useEffect(() => {
    isMounted.current = true;
    refreshBackendStatus();
    return () => {
      isMounted.current = false;
    };
  }, [refreshBackendStatus]);

  /**
   * Send a user message to the FastAPI RAG backend.
   */
  const sendMessage = useCallback(async (text) => {
    const trimmed = (text || '').trim();
    if (!trimmed || isLoading) return;

    setError(null);
    setLastFailedMessage(null);

    const userMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: trimmed,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const rawResponse = await sendChatMessage(trimmed);
      const normalized = normalizeChatResponse(rawResponse);

      const assistantMessage = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: normalized.answer,
        sources: normalized.sources,
        hasWebFallback: normalized.hasWebFallback,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      if (isMounted.current) {
        setMessages((prev) => [...prev, assistantMessage]);
        setBackendStatus('online');
      }
    } catch (err) {
      if (isMounted.current) {
        setError(err.message || 'Failed to send message. Please try again.');
        setLastFailedMessage(trimmed);
        if (err.isNetworkError) {
          setBackendStatus('offline');
        }
      }
    } finally {
      if (isMounted.current) {
        setIsLoading(false);
      }
    }
  }, [isLoading]);

  /**
   * Retry the last question that failed to send.
   */
  const retryLastMessage = useCallback(() => {
    if (lastFailedMessage && !isLoading) {
      // Remove the last user message if we are retrying it to avoid duplication in display
      setMessages((prev) => {
        if (prev.length > 0 && prev[prev.length - 1].role === 'user') {
          return prev.slice(0, -1);
        }
        return prev;
      });
      sendMessage(lastFailedMessage);
    }
  }, [lastFailedMessage, isLoading, sendMessage]);

  /**
   * Clear all messages in the current session.
   */
  const clearChat = useCallback(() => {
    setMessages([]);
    setError(null);
    setLastFailedMessage(null);
  }, []);

  return {
    messages,
    isLoading,
    error,
    backendStatus,
    lastFailedMessage,
    sendMessage,
    retryLastMessage,
    clearChat,
    refreshBackendStatus,
  };
}

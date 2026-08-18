import React from 'react';
import { useChat } from '../../hooks/useChat';
import ChatHeader from './ChatHeader';
import ChatMessages from './ChatMessages';
import ChatInput from './ChatInput';
import ErrorMessage from './ErrorMessage';

export default function Chatbot() {
  const {
    messages,
    isLoading,
    error,
    backendStatus,
    lastFailedMessage,
    sendMessage,
    retryLastMessage,
    clearChat,
  } = useChat();

  return (
    <div className="chatbot-container" role="region" aria-label="AI Chatbot Assistant">
      <ChatHeader
        backendStatus={backendStatus}
        onClear={clearChat}
        messageCount={messages.length}
      />

      <ChatMessages
        messages={messages}
        isLoading={isLoading}
        onSelectSuggestion={(question) => sendMessage(question)}
      />

      {error && (
        <ErrorMessage
          message={error}
          onRetry={lastFailedMessage ? retryLastMessage : null}
        />
      )}

      <ChatInput
        onSend={(text) => sendMessage(text)}
        disabled={isLoading}
      />
    </div>
  );
}

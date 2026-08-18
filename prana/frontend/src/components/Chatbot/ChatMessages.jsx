import React, { useEffect, useRef } from 'react';
import { Sparkles, Bot } from 'lucide-react';
import ChatMessage from './ChatMessage';
import TypingIndicator from './TypingIndicator';
import SuggestedQuestions from './SuggestedQuestions';

export default function ChatMessages({ messages = [], isLoading = false, onSelectSuggestion }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  return (
    <div
      className="chat-messages-container"
      role="log"
      aria-label="Chat messages history"
      aria-live="polite"
    >
      {messages.length === 0 ? (
        <div className="welcome-card">
          <div className="welcome-icon-wrapper">
            <Sparkles size={28} />
          </div>
          <h2 className="welcome-title">Hi! How can I help you today?</h2>
          <p className="welcome-description">
            I am your dedicated website assistant powered by Retrieval-Augmented Generation. Ask me anything about our services, features, guides, or documentation.
          </p>

          <SuggestedQuestions onSelect={onSelectSuggestion} />
        </div>
      ) : (
        <>
          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))}

          {isLoading && (
            <div className="chat-message-row assistant">
              <div className="message-avatar assistant" aria-hidden="true">
                <Bot size={18} />
              </div>
              <div className="message-content-wrapper">
                <TypingIndicator />
              </div>
            </div>
          )}
        </>
      )}

      <div ref={bottomRef} style={{ height: '1px' }} />
    </div>
  );
}

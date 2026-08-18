import React from 'react';

export default function TypingIndicator() {
  return (
    <div
      className="typing-indicator"
      role="status"
      aria-live="polite"
      aria-label="Assistant is searching knowledge base and generating answer..."
    >
      <span className="typing-dot" />
      <span className="typing-dot" />
      <span className="typing-dot" />
    </div>
  );
}

import React, { useState, useRef, useEffect } from 'react';
import { Send, CornerDownLeft } from 'lucide-react';

const MAX_LENGTH = 2000;

export default function ChatInput({ onSend, disabled = false, placeholder = "Ask a question about the website..." }) {
  const [input, setInput] = useState('');
  const textareaRef = useRef(null);

  // Auto-resize textarea height based on content
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 140)}px`;
    }
  }, [input]);

  const handleSubmit = (e) => {
    if (e) e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || disabled) return;

    onSend(trimmed);
    setInput('');

    // Reset height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const isOverLimit = input.length > MAX_LENGTH;
  const canSubmit = input.trim().length > 0 && !disabled && !isOverLimit;

  return (
    <div className="chat-input-container">
      <form onSubmit={handleSubmit} className="chat-input-form">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          rows={1}
          maxLength={MAX_LENGTH + 50}
          className="chat-textarea"
          aria-label="Message input"
        />

        <button
          type="submit"
          disabled={!canSubmit}
          className="chat-send-btn"
          aria-label="Send message"
          title="Send (Enter)"
        >
          <Send size={16} />
        </button>
      </form>

      <div className="input-footer-info">
        <span className="input-hint">
          <CornerDownLeft size={11} />
          <span>Press <strong>Enter</strong> to send, <strong>Shift + Enter</strong> for new line</span>
        </span>

        {input.length > 100 && (
          <span style={{ color: isOverLimit ? 'var(--error-color)' : 'var(--text-muted)' }}>
            {input.length} / {MAX_LENGTH}
          </span>
        )}
      </div>
    </div>
  );
}

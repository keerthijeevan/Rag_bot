import React from 'react';
import { Bot, RotateCcw, Activity } from 'lucide-react';

export default function ChatHeader({ backendStatus, onClear, messageCount }) {
  const isOnline = backendStatus === 'online';
  const isChecking = backendStatus === 'checking';

  return (
    <header className="chat-header">
      <div className="chat-header-info">
        <div className="chat-header-avatar">
          <Bot size={22} />
        </div>
        <div>
          <h1 className="chat-header-title">
            Website AI Assistant
          </h1>
          <div className="chat-header-status">
            <span
              className={`status-dot ${isOnline ? 'online' : isChecking ? '' : 'offline'}`}
              aria-hidden="true"
            />
            <span>
              {isChecking ? 'Connecting...' : isOnline ? 'Online • Ready to help' : 'Service Offline'}
            </span>
          </div>
        </div>
      </div>

      <div className="chat-header-actions">
        {messageCount > 0 && (
          <button
            type="button"
            className="header-btn"
            onClick={onClear}
            title="Start new conversation"
            aria-label="Clear chat session"
          >
            <RotateCcw size={14} />
            <span>New Chat</span>
          </button>
        )}
      </div>
    </header>
  );
}

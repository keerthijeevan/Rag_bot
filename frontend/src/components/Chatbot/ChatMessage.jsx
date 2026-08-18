import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Bot, User, Globe } from 'lucide-react';
import SourceList from './SourceList';

export default function ChatMessage({ message }) {
  const isUser = message.role === 'user';

  return (
    <div className={`chat-message-row ${isUser ? 'user' : 'assistant'}`}>
      <div
        className={`message-avatar ${isUser ? 'user' : 'assistant'}`}
        aria-hidden="true"
      >
        {isUser ? <User size={18} /> : <Bot size={18} />}
      </div>

      <div className="message-content-wrapper">
        <div className={`message-bubble ${isUser ? 'user' : 'assistant'}`}>
          {isUser ? (
            <div style={{ whiteSpace: 'pre-wrap' }}>{message.content}</div>
          ) : (
            <>
              {message.hasWebFallback && (
                <div className="web-fallback-badge">
                  <Globe size={13} />
                  <span>Information supplemented from website</span>
                </div>
              )}

              <div className="markdown-body">
                <ReactMarkdown>{message.content}</ReactMarkdown>
              </div>

              {message.sources && message.sources.length > 0 && (
                <SourceList sources={message.sources} />
              )}
            </>
          )}
        </div>

        {message.timestamp && (
          <div
            style={{
              fontSize: '0.72rem',
              color: 'var(--text-muted)',
              alignSelf: isUser ? 'flex-end' : 'flex-start',
              padding: '0 4px',
            }}
          >
            {message.timestamp}
          </div>
        )}
      </div>
    </div>
  );
}

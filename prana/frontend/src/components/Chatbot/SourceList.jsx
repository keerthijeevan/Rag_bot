import React, { useState } from 'react';
import { BookOpen, ChevronDown, ChevronUp, ExternalLink, FileText } from 'lucide-react';
import { getSourceTypeLabel } from '../../utils/responseParser';

export default function SourceList({ sources = [] }) {
  const [isOpen, setIsOpen] = useState(false);

  if (!sources || sources.length === 0) {
    return null;
  }

  return (
    <div className="sources-container">
      <button
        type="button"
        className="sources-toggle-btn"
        onClick={() => setIsOpen((prev) => !prev)}
        aria-expanded={isOpen}
        aria-label={`Toggle sources list, ${sources.length} sources available`}
      >
        <BookOpen size={14} />
        <span>Sources ({sources.length})</span>
        {isOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>

      {isOpen && (
        <div className="sources-list" role="list">
          {sources.map((source, index) => {
            const hasUrl = Boolean(source.url);
            const badgeLabel = getSourceTypeLabel(source.sourceType);

            const content = (
              <>
                <div className="source-item-left">
                  <span className="source-badge">{badgeLabel}</span>
                  <span className="source-title" title={source.title}>
                    {source.title}
                  </span>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  {source.page && (
                    <span className="source-page">Page {source.page}</span>
                  )}
                  {hasUrl && <ExternalLink size={13} style={{ color: 'var(--text-muted)' }} />}
                </div>
              </>
            );

            if (hasUrl) {
              return (
                <a
                  key={source.id || index}
                  href={source.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="source-item"
                  role="listitem"
                  title={`Open external source: ${source.title}`}
                >
                  {content}
                </a>
              );
            }

            return (
              <div
                key={source.id || index}
                className="source-item"
                role="listitem"
              >
                {content}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

import React from 'react';
import { HelpCircle } from 'lucide-react';

const DEFAULT_SUGGESTIONS = [
  "What services does the website provide?",
  "What are your key features and capabilities?",
  "How can I get in touch or contact support?",
  "What documentation or resources are available?"
];

export default function SuggestedQuestions({ onSelect, suggestions = DEFAULT_SUGGESTIONS }) {
  if (!suggestions || suggestions.length === 0) return null;

  return (
    <div className="suggested-questions">
      <div className="suggested-title">
        <span>Suggested Questions</span>
      </div>
      <div className="suggested-grid">
        {suggestions.map((question, index) => (
          <button
            key={index}
            type="button"
            className="suggestion-chip"
            onClick={() => onSelect(question)}
            aria-label={`Ask suggested question: ${question}`}
          >
            {question}
          </button>
        ))}
      </div>
    </div>
  );
}

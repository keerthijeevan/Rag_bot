# Website AI RAG Chatbot - Frontend

Modern, responsive React frontend interface for the Website-specific AI RAG Chatbot. Integrates directly with the FastAPI RAG backend (`/rag/chat` and `/health`).

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Node.js**: v18+ (tested on Node.js v24.x LTS)
- **FastAPI Backend**: Running on `http://localhost:8000`

### 2. Setup & Configuration

From the project root:

```bash
cd frontend
cp .env.example .env
npm install
```

Ensure `.env` points to your running FastAPI backend:
```env
VITE_RAG_API_URL=http://localhost:8000
```

### 3. Run Development Server

```bash
npm run dev
```

Open your browser at:
**[http://localhost:5173](http://localhost:5173)**

---

## 🧪 Testing & Building

```bash
# Run unit & component tests
npm test

# Build optimized production bundle
npm run build

# Preview production build locally
npm run preview
```

---

## 🏗 Architecture & Flow

```
[ Browser / React Frontend ]
            │
   (HTTP POST /rag/chat)
            │
            ▼
   [ FastAPI Backend ]
            │
   [ LangGraph Controller ]
      ├── Message Classifier (GREETING / OUT_OF_SCOPE / KNOWLEDGE_QUERY)
      ├── Hybrid Retrieval (Dense BGE + Sparse BM25 via Qdrant)
      ├── BGE Reranker
      ├── Optional Web Fallback Scraper
      └── Gemini Answer Generation
            │
            ▼
[ Chat Response: Answer + Sources + Metadata ]
            │
            ▼
[ React UI: Markdown Formatting + Source Cards + Safe Links ]
```

---

## 📁 Directory Structure

```
frontend/
├── src/
│   ├── components/Chatbot/
│   │   ├── Chatbot.jsx            # Main container component
│   │   ├── ChatHeader.jsx         # Header with live status & reset button
│   │   ├── ChatMessages.jsx       # Message stream & welcome card
│   │   ├── ChatMessage.jsx        # Message bubble with Markdown formatting
│   │   ├── SourceList.jsx         # Expandable sources accordion & external links
│   │   ├── SuggestedQuestions.jsx # Starter chips for common queries
│   │   ├── TypingIndicator.jsx    # Animated pulsing typing indicator
│   │   ├── ChatInput.jsx          # Auto-resizing input with keyboard shortcuts
│   │   └── ErrorMessage.jsx       # User-friendly error alert with Retry action
│   ├── hooks/
│   │   └── useChat.js             # Session state & API communication hook
│   ├── services/
│   │   └── ragApi.js              # HTTP client with timeouts & error normalization
│   ├── utils/
│   │   └── responseParser.js      # Parser for normalizing answer, sources & badges
│   ├── styles/
│   │   ├── index.css              # Global tokens & CSS variables
│   │   └── chatbot.css            # Chatbot styling & responsive media queries
│   ├── App.jsx
│   └── main.jsx
├── .env.example
├── package.json
└── vite.config.js
```

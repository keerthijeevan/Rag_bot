# Website AI RAG Chatbot – Frontend

Modern, responsive React frontend for the Website AI RAG Chatbot. It connects directly to the FastAPI backend through:

- `POST /rag/chat`
- `GET /rith`

## Quick Start Guide

### Prerequisites

- Node.js 18 or later
- FastAPI backend running at `http://localhost:8000`

### Setup and configuration

From the project root:

```powershell
cd frontend
Copy-Item .env.example .env
npm install
```

Ensure `.env` contains:

```env
VITE_RAG_API_URL=http://localhost:8000
```

### Run the development server

```powershell
npm run dev
```

Open:

[http://localhost:3000](http://localhost:3000)

> `vite.config.js` must use port `3000` so it matches the backend's permitted local origin.

## Testing and Building

```powershell
# Run unit and component tests
npm test

# Build the production bundle
npm run build

# Preview the production build locally
npm run preview
```

## Architecture and Flow

```text
[ Browser / React Frontend ]
            |
     HTTP POST /rag/chat
            |
            v
   [ FastAPI Backend ]
            |
   [ LangGraph Controller ]
      |- Greeting Classifier (GREETING / NOT_GREETING)
      |- Hybrid Retrieval (Dense BGE + Sparse BM25 via Qdrant)
      |- BGE Reranker
      |- Optional Website Fallback
      `- Gemini Answer Generation
            |
            v
     [ Chat Response: Answer ]
            |
            v
[ React UI: Markdown Formatting + Safe Links ]
```

> The frontend supports source cards. They will appear when the backend returns a `sources` array in its chat response.

## Directory Structure

```text
frontend/
├── src/
│   ├── components/Chatbot/
│   │   ├── Chatbot.jsx            # Main chatbot container
│   │   ├── ChatHeader.jsx         # Header with service status and reset button
│   │   ├── ChatMessages.jsx       # Message stream and welcome screen
│   │   ├── ChatMessage.jsx        # Message bubble with Markdown rendering
│   │   ├── SourceList.jsx         # Expandable source list and external links
│   │   ├── SuggestedQuestions.jsx # Starter questions
│   │   ├── TypingIndicator.jsx    # Loading animation
│   │   ├── ChatInput.jsx          # Auto-resizing message input
│   │   └── ErrorMessage.jsx       # Error display and retry action
│   ├── hooks/
│   │   └── useChat.js             # Chat state and API communication
│   ├── services/
│   │   └── ragApi.js              # FastAPI HTTP client and error handling
│   ├── utils/
│   │   └── responseParser.js      # Response normalization and source labels
│   ├── styles/
│   │   ├── index.css              # Global styles and CSS variables
│   │   └── chatbot.css            # Chatbot layout and responsive styles
│   ├── App.jsx
│   └── main.jsx
├── .env.example
├── package.json
└── vite.config.js
```
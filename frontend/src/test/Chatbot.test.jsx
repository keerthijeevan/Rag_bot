import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import Chatbot from '../components/Chatbot/Chatbot';
import * as ragApi from '../services/ragApi';

vi.mock('../services/ragApi', async () => {
  const actual = await vi.importActual('../services/ragApi');
  return {
    ...actual,
    sendChatMessage: vi.fn(),
    checkHealth: vi.fn(),
  };
});

describe('Chatbot component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    ragApi.checkHealth.mockResolvedValue({ online: true, service: 'rag-fastapi' });
  });

  it('renders welcome screen and suggestions initially', async () => {
    render(<Chatbot />);
    
    expect(screen.getByText(/Website AI Assistant/i)).toBeInTheDocument();
    expect(screen.getByText(/Hi! How can I help you today\?/i)).toBeInTheDocument();
    expect(screen.getByText(/What services does the website provide\?/i)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText(/Online • Ready to help/i)).toBeInTheDocument();
    });
  });

  it('sends a question and renders assistant answer and sources', async () => {
    ragApi.sendChatMessage.mockResolvedValue({
      answer: 'We provide RAG pipeline integrations.',
      sources: [
        {
          title: 'Services Guide',
          source_type: 'pdf',
          url: 'https://example.com/guide.pdf',
          page: 1
        }
      ]
    });

    render(<Chatbot />);

    const textarea = screen.getByLabelText(/Message input/i);
    const sendButton = screen.getByLabelText(/Send message/i);

    fireEvent.change(textarea, { target: { value: 'What services do you offer?' } });
    fireEvent.click(sendButton);

    // User message should appear
    expect(screen.getByText('What services do you offer?')).toBeInTheDocument();

    // Assistant response should appear
    await waitFor(() => {
      expect(screen.getByText('We provide RAG pipeline integrations.')).toBeInTheDocument();
    });

    // Sources button should exist
    expect(screen.getByText(/Sources \(1\)/i)).toBeInTheDocument();
  });

  it('displays error banner and allows retry when API fails', async () => {
    ragApi.sendChatMessage.mockRejectedValueOnce(
      new Error('Unable to connect to the chatbot service.')
    );

    render(<Chatbot />);

    const textarea = screen.getByLabelText(/Message input/i);
    fireEvent.change(textarea, { target: { value: 'Test query' } });
    fireEvent.submit(textarea.closest('form'));

    await waitFor(() => {
      expect(screen.getByText(/Unable to connect to the chatbot service/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Retry/i })).toBeInTheDocument();
    });

    // Mock successful retry
    ragApi.sendChatMessage.mockResolvedValueOnce({
      answer: 'Success after retry!',
      sources: []
    });

    fireEvent.click(screen.getByRole('button', { name: /Retry/i }));

    await waitFor(() => {
      expect(screen.getByText('Success after retry!')).toBeInTheDocument();
    });
  });
});

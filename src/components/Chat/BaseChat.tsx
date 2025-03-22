import React, { useState, useEffect, useRef } from 'react';

interface BaseChatProps {
  chatHistory: Array<{
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
  }>;
  className?: string;
  chatContainerRef: React.RefObject<HTMLDivElement>;
  onSubmit: (message: string) => void;
  isLoading: boolean;
  error: string | null;
}

const BaseChat: React.FC<BaseChatProps> = ({
  chatHistory,
  className,
  chatContainerRef,
  onSubmit,
  isLoading,
  error
}) => {
  const [message, setMessage] = useState('');
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Handle submit on enter key
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (message.trim() && !isLoading) {
        onSubmit(message);
        setMessage('');
      }
    }
  };

  // Submit form handler
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !isLoading) {
      onSubmit(message);
      setMessage('');
    }
  };

  // Auto-resize textarea
  useEffect(() => {
    const textarea = inputRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${textarea.scrollHeight}px`;
    }
  }, [message]);

  // Scroll to bottom when chat history changes
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatHistory, chatContainerRef]);

  return (
    <div className={`chat-container ${className || ''}`}>
      <div 
        ref={chatContainerRef}
        className="messages-container"
        style={{ 
          height: '70vh', 
          overflowY: 'auto',
          padding: '1rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '1rem'
        }}
      >
        {chatHistory.map((msg, index) => (
          <div 
            key={index} 
            className={`message ${msg.role === 'user' ? 'user-message' : 'assistant-message'}`}
            style={{
              padding: '0.75rem 1rem',
              borderRadius: '0.5rem',
              maxWidth: '80%',
              alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
              backgroundColor: msg.role === 'user' ? '#1e88e5' : '#f5f5f5',
              color: msg.role === 'user' ? 'white' : 'black'
            }}
          >
            <div className="message-content">{msg.content}</div>
            <div 
              className="message-timestamp"
              style={{
                fontSize: '0.75rem',
                opacity: 0.7,
                marginTop: '0.25rem',
                textAlign: msg.role === 'user' ? 'right' : 'left'
              }}
            >
              {msg.timestamp.toLocaleTimeString()}
            </div>
          </div>
        ))}
        {isLoading && (
          <div 
            className="loading-indicator"
            style={{
              padding: '0.75rem 1rem',
              borderRadius: '0.5rem',
              alignSelf: 'flex-start',
              backgroundColor: '#f5f5f5'
            }}
          >
            Thinking...
          </div>
        )}
        {error && (
          <div 
            className="error-message"
            style={{
              padding: '0.75rem 1rem',
              borderRadius: '0.5rem',
              alignSelf: 'center',
              backgroundColor: '#ffebee',
              color: '#d32f2f'
            }}
          >
            {error}
          </div>
        )}
      </div>
      
      <form 
        onSubmit={handleSubmit}
        style={{
          display: 'flex',
          gap: '0.5rem',
          padding: '1rem',
          borderTop: '1px solid #eee'
        }}
      >
        <textarea
          ref={inputRef}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message here..."
          style={{
            flex: 1,
            padding: '0.75rem',
            borderRadius: '0.5rem',
            border: '1px solid #ddd',
            resize: 'none',
            minHeight: '2.5rem',
            maxHeight: '10rem',
            overflowY: 'auto'
          }}
        />
        <button
          type="submit"
          disabled={isLoading || !message.trim()}
          style={{
            padding: '0.75rem 1.5rem',
            borderRadius: '0.5rem',
            backgroundColor: isLoading || !message.trim() ? '#ccc' : '#1e88e5',
            color: 'white',
            border: 'none',
            cursor: isLoading || !message.trim() ? 'not-allowed' : 'pointer'
          }}
        >
          Send
        </button>
      </form>
    </div>
  );
};

export default BaseChat; 
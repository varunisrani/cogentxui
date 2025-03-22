import React, { useState, useRef, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import BaseChat from './BaseChat';
import { useChatHistory } from './hooks/useChatHistory';
import { useWebSocketChat } from './hooks/useWebSocketChat';

const API_URL = 'http://localhost:8001';

interface ChatProps {
  className?: string;
}

const Chat: React.FC<ChatProps> = ({ className }) => {
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const initialPrompt = queryParams.get('prompt') || '';
  
  const { chatHistory, addUserMessage, addBotMessage, updateBotMessage, setChatHistory } = useChatHistory();
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [currentThreadId, setCurrentThreadId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  // Custom hook for WebSocket connection
  const { 
    isConnected, 
    connect, 
    disconnect, 
    sendMessage 
  } = useWebSocketChat({
    onMessage: (chunk: string) => {
      updateBotMessage(prevMessage => prevMessage + chunk);
    },
    onComplete: () => {
      setIsLoading(false);
    },
    onError: (err: string) => {
      setError(err);
      setIsLoading(false);
    }
  });

  useEffect(() => {
    // Submit initial prompt if provided via URL
    if (initialPrompt && chatHistory.length === 0) {
      handleSubmit(initialPrompt);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialPrompt]);

  // Scroll to bottom when chat history changes
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatHistory]);

  // Check API connection on component mount
  useEffect(() => {
    const checkConnection = async () => {
      try {
        const response = await fetch(`${API_URL}/api/confirm`);
        const data = await response.json();
        console.log('API connection:', data.connected ? 'Successful' : 'Failed');
        setError(data.connected ? null : 'API connection failed');
      } catch (err) {
        console.error('Failed to connect to API:', err);
        setError('Failed to connect to API server. Please make sure the backend is running.');
      }
    };
    
    checkConnection();
  }, []);

  const handleSubmit = async (message: string) => {
    if (!message.trim() || isLoading) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      // Add user message to chat
      addUserMessage(message);
      
      if (isConnected) {
        // If already connected to WebSocket, send message through it
        sendMessage(message, currentThreadId !== null);
      } else {
        // For first message, use HTTP endpoint to get thread_id
        const response = await fetch(`${API_URL}/api/chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message,
            thread_id: currentThreadId,
          }),
        });
        
        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.thread_id && data.thread_id !== currentThreadId) {
          setCurrentThreadId(data.thread_id);
          // Connect to WebSocket for streaming future messages
          await connect(data.thread_id);
        }
        
        // Add bot response to chat
        addBotMessage(data.response);
        setIsLoading(false);
      }
    } catch (err: any) {
      console.error('Error submitting message:', err);
      setError(err.message || 'Failed to send message');
      setIsLoading(false);
    }
  };

  return (
    <BaseChat
      chatHistory={chatHistory}
      className={className}
      chatContainerRef={chatContainerRef}
      onSubmit={handleSubmit}
      isLoading={isLoading}
      error={error}
    />
  );
};

export default Chat; 
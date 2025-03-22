import { useState, useCallback } from 'react';

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export const useChatHistory = () => {
  const [chatHistory, setChatHistory] = useState<Message[]>([]);

  // Add a user message to the chat history
  const addUserMessage = useCallback((content: string) => {
    setChatHistory(prevHistory => [
      ...prevHistory,
      {
        role: 'user',
        content,
        timestamp: new Date()
      }
    ]);
  }, []);

  // Add a bot message to the chat history
  const addBotMessage = useCallback((content: string) => {
    setChatHistory(prevHistory => [
      ...prevHistory,
      {
        role: 'assistant',
        content,
        timestamp: new Date()
      }
    ]);
  }, []);

  // Function to update the most recent bot message for streaming
  const updateBotMessage = useCallback((updater: (prev: string) => string) => {
    setChatHistory(prevHistory => {
      // Find the last bot message, or add a new one if there isn't one
      const lastBotIndex = [...prevHistory].reverse().findIndex(msg => msg.role === 'assistant');
      
      if (lastBotIndex === -1) {
        // No bot message found, add new one
        return [
          ...prevHistory,
          {
            role: 'assistant',
            content: updater(''),
            timestamp: new Date()
          }
        ];
      } else {
        // Update the last bot message
        const actualIndex = prevHistory.length - 1 - lastBotIndex;
        const newHistory = [...prevHistory];
        const oldContent = newHistory[actualIndex].content;
        newHistory[actualIndex] = {
          ...newHistory[actualIndex],
          content: updater(oldContent)
        };
        return newHistory;
      }
    });
  }, []);

  // Clear the chat history
  const clearChatHistory = useCallback(() => {
    setChatHistory([]);
  }, []);

  return {
    chatHistory,
    setChatHistory,
    addUserMessage,
    addBotMessage,
    updateBotMessage,
    clearChatHistory
  };
}; 
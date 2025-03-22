import { useState, useCallback, useEffect } from 'react';
import type { Message } from 'ai';

interface UseChatOptions {
  initialMessages?: Message[];
  initialInput?: string;
  onError?: (error: Error) => void;
  onFinish?: (message: Message, response: any) => void;
  body?: Record<string, any>;
  sendExtraMessageFields?: boolean;
}

// Define response types to fix type checking issues
interface ChatResponse {
  response: string;
  thread_id: string;
  success: boolean;
  is_first_message?: boolean;
  usage?: {
    completionTokens?: number;
    promptTokens?: number;
    totalTokens?: number;
  };
}

export function useCustomChat({
  initialMessages = [],
  initialInput = '',
  onError,
  onFinish,
  body = {},
}: UseChatOptions) {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [input, setInput] = useState(initialInput);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [controller, setController] = useState<AbortController | null>(null);
  const [data, setData] = useState<any[] | undefined>(undefined);

  // Function to handle input changes
  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      setInput(e.target.value);
    },
    []
  );

  // Function to stop ongoing requests
  const stop = useCallback(() => {
    if (controller) {
      controller.abort();
      setIsLoading(false);
    }
  }, [controller]);

  // Function to send a message using the HTTP endpoint
  const append = useCallback(
    async (message: Message) => {
      if (isLoading) {
        return;
      }

      setIsLoading(true);
      const abortController = new AbortController();
      setController(abortController);

      try {
        // Extract the actual text content from the message
        let messageText = message.content;
        if (Array.isArray(message.content)) {
          // Handle complex content structure (with text and images)
          const textContent = message.content.find((item: any) => item.type === 'text');
          messageText = textContent ? textContent.text : '';
        }
        
        // Create a thread ID from existing messages if available
        const thread_id = messages.length > 0 && messages[0].id ? messages[0].id : undefined;
        
        // Make the API request
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message: messageText,
            thread_id,
            ...body,
          }),
          signal: abortController.signal,
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error((errorData as any)?.detail || `Request failed with status ${response.status}`);
        }

        const responseData = await response.json() as ChatResponse;
        
        // Create new messages array with user message and response
        const newMessages: Message[] = [
          ...messages,
          message, // User message
          {
            id: responseData.thread_id || Date.now().toString(),
            role: 'assistant',
            content: responseData.response,
          } as Message, // Assistant response
        ];

        setMessages(newMessages);
        
        // Call onFinish callback if provided
        if (onFinish) {
          onFinish(newMessages[newMessages.length - 1], responseData);
        }
      } catch (e) {
        const errorAsError = e as Error;
        setError(errorAsError);
        if (onError) {
          onError(errorAsError);
        }
      } finally {
        setIsLoading(false);
        setController(null);
        setInput('');
      }
    },
    [messages, isLoading, body, onError, onFinish]
  );

  // Function to reload the last user message
  const reload = useCallback(() => {
    if (messages.length === 0) return;
    
    // Find the last user message
    const lastUserMessageIndex = [...messages].reverse().findIndex(m => m.role === 'user');
    if (lastUserMessageIndex === -1) return;
    
    const lastUserMessage = messages[messages.length - 1 - lastUserMessageIndex];
    
    // Remove all messages after the last user message
    const newMessages = messages.slice(0, messages.length - lastUserMessageIndex);
    setMessages(newMessages);
    
    // Re-send the last user message
    append(lastUserMessage);
  }, [messages, append]);

  return {
    messages,
    input,
    handleInputChange,
    setInput,
    isLoading,
    error,
    append,
    stop,
    reload,
    setMessages,
    data,
    setData
  };
}

// Websocket-based streaming chat hook
export function useStreamingChat({
  initialMessages = [],
  initialInput = '',
  onError,
  onFinish,
  body = {},
}: UseChatOptions) {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [input, setInput] = useState(initialInput);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [data, setData] = useState<any[] | undefined>(undefined);
  const [streamedContent, setStreamedContent] = useState<string>('');

  // Function to handle input changes
  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      setInput(e.target.value);
    },
    []
  );

  // Function to stop the stream
  const stop = useCallback(() => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.close();
      setIsLoading(false);
    }
  }, [socket]);

  // Create a new websocket connection with a thread_id
  const createWebSocketConnection = useCallback((thread_id: string) => {
    // For development, directly use the FastAPI server URL
    const wsUrl = `ws://localhost:8001/ws/chat/${thread_id}`;
    
    console.log('Connecting to WebSocket URL:', wsUrl);
    
    const ws = new WebSocket(wsUrl);
    
    // Add error handling
    ws.onerror = (event) => {
      console.error('WebSocket connection error:', event);
      // Try with fallback to current host if direct connection fails
      const fallbackProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const fallbackHost = window.location.host;
      const fallbackUrl = `${fallbackProtocol}//${fallbackHost}/ws/chat/${thread_id}`;
      
      console.log('Retrying with fallback WebSocket URL:', fallbackUrl);
      
      const fallbackWs = new WebSocket(fallbackUrl);
      setSocket(fallbackWs);
      return fallbackWs;
    };
    
    setSocket(ws);
    return ws;
  }, []);

  // Function to send a message
  const append = useCallback(
    async (userMessage: Message) => {
      if (isLoading) {
        stop();
        return;
      }

      setIsLoading(true);
      setStreamedContent('');

      try {
        // Extract the actual text content from the message
        let messageText = userMessage.content;
        if (Array.isArray(userMessage.content)) {
          // Handle complex content structure (with text and images)
          const textContent = userMessage.content.find((item: any) => item.type === 'text');
          messageText = textContent ? textContent.text : '';
        }

        // Determine if this is the first message
        const isFirstMessage = messages.length === 0;
        
        // Create a thread_id - use the existing one or generate a new one for first message
        const thread_id = !isFirstMessage && messages[0].id 
          ? messages[0].id 
          : crypto.randomUUID();

        // Create a message with an ID if it doesn't have one
        const messageWithId: Message = {
          ...userMessage,
          id: userMessage.id || thread_id
        };

        // Add user message to the list
        const updatedMessages = [...messages, messageWithId];
        setMessages(updatedMessages);

        // Initialize WebSocket
        const ws = createWebSocketConnection(thread_id);

        ws.onopen = () => {
          // Send the message when the socket is open
          ws.send(JSON.stringify({
            message: messageText,
            is_first_message: isFirstMessage,
            ...body,
          }));
        };

        let responseMessage: Message = {
          id: `response-${Date.now()}`,
          role: 'assistant',
          content: '',
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            
            if (data.type === 'error') {
              throw new Error(data.content);
            } else if (data.type === 'chunk') {
              // Update the streamed content
              setStreamedContent(prev => prev + data.content);
              
              // Update the response message
              responseMessage = {
                ...responseMessage,
                content: responseMessage.content + data.content,
              };
              
              // Update messages with the latest content
              setMessages([...updatedMessages, responseMessage]);
            } else if (data.type === 'complete') {
              setIsLoading(false);
              
              // Call onFinish callback if provided
              if (onFinish) {
                onFinish(responseMessage, { 
                  usage: null,
                  thread_id,
                  success: true
                });
              }
              
              ws.close();
            }
          } catch (e) {
            const errorAsError = e as Error;
            setError(errorAsError);
            if (onError) {
              onError(errorAsError);
            }
            ws.close();
          }
        };

        ws.onerror = (event) => {
          console.error('WebSocket error:', event);
          const error = new Error('WebSocket connection error');
          setError(error);
          if (onError) {
            onError(error);
          }
          setIsLoading(false);
        };

        ws.onclose = () => {
          setIsLoading(false);
          setSocket(null);
        };
      } catch (e) {
        const errorAsError = e as Error;
        setError(errorAsError);
        if (onError) {
          onError(errorAsError);
        }
        setIsLoading(false);
      }
    },
    [messages, isLoading, body, onError, onFinish, createWebSocketConnection, stop]
  );

  // Function to reload the last user message
  const reload = useCallback(() => {
    if (messages.length === 0) return;
    
    // Find the last user message
    const lastUserMessageIndex = [...messages].reverse().findIndex(m => m.role === 'user');
    if (lastUserMessageIndex === -1) return;
    
    const lastUserMessage = messages[messages.length - 1 - lastUserMessageIndex];
    
    // Remove all messages after the last user message
    const newMessages = messages.slice(0, messages.length - lastUserMessageIndex);
    setMessages(newMessages);
    
    // Re-send the last user message
    append(lastUserMessage);
  }, [messages, append]);

  // Clean up the WebSocket connection when component unmounts
  useEffect(() => {
    return () => {
      if (socket) {
        socket.close();
      }
    };
  }, [socket]);

  return {
    messages,
    input,
    handleInputChange,
    setInput,
    isLoading,
    error,
    append,
    stop,
    reload,
    setMessages,
    data,
    setData
  };
} 
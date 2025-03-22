import { useCallback, useEffect, useRef, useState } from 'react';

const API_URL = 'http://localhost:8001';

interface WebSocketHookOptions {
  onMessage: (chunk: string) => void;
  onComplete: () => void;
  onError: (err: string) => void;
}

export const useWebSocketChat = (options: WebSocketHookOptions) => {
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const webSocketRef = useRef<WebSocket | null>(null);
  const threadIdRef = useRef<string | null>(null);

  // Clean up the WebSocket connection when the component unmounts
  useEffect(() => {
    return () => {
      if (webSocketRef.current) {
        webSocketRef.current.close();
        webSocketRef.current = null;
        setIsConnected(false);
      }
    };
  }, []);

  // Function to connect to the WebSocket
  const connect = useCallback(
    async (threadId: string) => {
      if (isConnected || isConnecting) {
        return;
      }

      setIsConnecting(true);
      threadIdRef.current = threadId;

      try {
        // Ensure we close any existing connection
        if (webSocketRef.current) {
          webSocketRef.current.close();
        }

        // Create new WebSocket connection
        const wsUrl = `ws://${API_URL.replace('http://', '')}/ws/chat/${threadId}`;
        console.log(`Connecting to WebSocket at: ${wsUrl}`);
        
        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
          console.log(`WebSocket connected for thread: ${threadId}`);
          setIsConnected(true);
          setIsConnecting(false);
          webSocketRef.current = ws;
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            
            if (data.type === 'chunk') {
              options.onMessage(data.content);
            } else if (data.type === 'complete') {
              options.onComplete();
            } else if (data.type === 'error') {
              options.onError(data.content);
            }
          } catch (err) {
            console.error('Failed to parse WebSocket message:', err, event.data);
            options.onError('Failed to parse server response');
          }
        };

        ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          options.onError('WebSocket connection error');
          setIsConnected(false);
          setIsConnecting(false);
          webSocketRef.current = null;
        };

        ws.onclose = () => {
          console.log('WebSocket closed');
          setIsConnected(false);
          setIsConnecting(false);
          webSocketRef.current = null;
        };

        // Wait for the connection to be established or failed
        await new Promise<void>((resolve, reject) => {
          const timeoutId = setTimeout(() => {
            reject(new Error('WebSocket connection timeout'));
          }, 5000);

          ws.onopen = () => {
            clearTimeout(timeoutId);
            setIsConnected(true);
            setIsConnecting(false);
            webSocketRef.current = ws;
            resolve();
          };

          ws.onerror = () => {
            clearTimeout(timeoutId);
            setIsConnected(false);
            setIsConnecting(false);
            reject(new Error('WebSocket connection error'));
          };
        });
      } catch (err) {
        console.error('Failed to connect to WebSocket:', err);
        options.onError('Failed to connect to chat server');
        setIsConnected(false);
        setIsConnecting(false);
        return false;
      }

      return true;
    },
    [isConnected, isConnecting, options]
  );

  // Function to disconnect from the WebSocket
  const disconnect = useCallback(() => {
    if (webSocketRef.current) {
      webSocketRef.current.close();
      webSocketRef.current = null;
      setIsConnected(false);
    }
  }, []);

  // Function to send a message via the WebSocket
  const sendMessage = useCallback(
    (message: string, isContinuation: boolean) => {
      if (!webSocketRef.current || webSocketRef.current.readyState !== WebSocket.OPEN) {
        options.onError('WebSocket not connected');
        return false;
      }

      try {
        webSocketRef.current.send(
          JSON.stringify({
            message,
            thread_id: threadIdRef.current,
            is_first_message: !isContinuation
          })
        );
        return true;
      } catch (err) {
        console.error('Failed to send WebSocket message:', err);
        options.onError('Failed to send message to chat server');
        return false;
      }
    },
    [options]
  );

  return {
    isConnected,
    isConnecting,
    connect,
    disconnect,
    sendMessage,
  };
}; 
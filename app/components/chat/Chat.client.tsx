/**
 * @ts-nocheck
 * Preventing TS checks with files presented in the video for a better presentation.
 */
import { useStore } from '@nanostores/react';
import type { Message } from 'ai';
import { useAnimate } from 'framer-motion';
import { memo, useCallback, useEffect, useRef, useState, Suspense } from 'react';
import { toast, ToastContainer } from 'react-toastify';
import { useMessageParser, useShortcuts, useSnapScroll } from '~/lib/hooks';
import { description, useChatHistory } from '~/lib/persistence';
import { chatStore } from '~/lib/stores/chat';
import { workbenchStore } from '~/lib/stores/workbench';
import { PROMPT_COOKIE_KEY, SHOW_RAW_API_RESPONSES } from '~/utils/constants';
import { cubicEasingFn } from '~/utils/easings';
import { createScopedLogger, renderLogger } from '~/utils/logger';
import { BaseChat } from './BaseChat';
import Cookies from 'js-cookie';
import { debounce } from '~/utils/debounce';
import { useSettings } from '~/lib/hooks/useSettings';
import { useSearchParams } from '@remix-run/react';
import { createSampler } from '~/utils/sampler';
import { logStore } from '~/lib/stores/logs';
import { streamingState } from '~/lib/stores/streaming';
import { filesToArtifacts } from '~/utils/fileUtils';
import { useStreamingChat } from '~/lib/hooks/useCustomChat';
import { ScreenshotStateManager } from './ScreenshotStateManager';

// API URL for our backend
const API_URL = 'http://localhost:8001';

const logger = createScopedLogger('Chat');

// Define types for API responses
interface ApiHealthResponse {
  status: string;
  components?: {
    api: boolean;
  };
  active_connections?: number;
  timestamp?: number;
}

interface ApiChatResponse {
  response: string;
  thread_id: string;
  success: boolean;
  is_first_message?: boolean;
}

// Simple inline debug panel component that won't cause rendering issues
function SimpleDebugPanel() {
  if (!SHOW_RAW_API_RESPONSES) return null;
  
  try {
    const debugMessages = useStore(workbenchStore.debugMessages);
    if (!debugMessages || debugMessages.length === 0) return null;
    
    return (
      <div className="fixed bottom-4 right-4 z-50 max-w-md max-h-96 overflow-auto bg-black bg-opacity-90 text-white rounded-lg p-4 border border-gray-700 text-xs">
        <div className="flex justify-between items-center mb-2">
          <h3 className="font-bold">API Debug Information</h3>
          <button onClick={() => workbenchStore.debugMessages.set([])} className="text-gray-400 hover:text-white">
            Clear
          </button>
        </div>
        <div className="space-y-2">
          {debugMessages.map((message, index) => (
            <div key={index} className={`p-2 rounded ${message.type === 'api-error' ? 'bg-red-900 bg-opacity-40' : 'bg-gray-800'}`}>
              <div className="font-semibold text-xs mb-1">{message.type} - {message.timestamp ? new Date(message.timestamp).toLocaleTimeString() : 'No timestamp'}</div>
              <pre className="whitespace-pre-wrap text-xs overflow-x-auto">{message.content}</pre>
            </div>
          ))}
        </div>
      </div>
    );
  } catch (error) {
    console.error("Error rendering debug panel:", error);
    return null;
  }
}

export function Chat() {
  renderLogger.trace('Chat');

  const { ready, initialMessages, storeMessageHistory, importChat, exportChat } = useChatHistory();
  const title = useStore(description);
  
  // Initialize debugMessages if not already initialized
  useEffect(() => {
    // Make sure debugMessages is initialized
    if (!workbenchStore.debugMessages.get()) {
      workbenchStore.debugMessages.set([]);
    }
    
    workbenchStore.setReloadedMessages(initialMessages.map((m) => m.id));
  }, [initialMessages]);

  return (
    <>
      {ready && (
        <ChatImpl
          description={title}
          initialMessages={initialMessages}
          exportChat={exportChat}
          storeMessageHistory={storeMessageHistory}
          importChat={importChat}
        />
      )}
      <ToastContainer
        closeButton={({ closeToast }) => {
          return (
            <button className="Toastify__close-button" onClick={closeToast}>
              <div className="i-ph:x text-lg" />
            </button>
          );
        }}
        icon={({ type }) => {
          switch (type) {
            case 'success': {
              return <div className="i-ph:check-bold text-bolt-elements-icon-success text-2xl" />;
            }
            case 'error': {
              return <div className="i-ph:warning-circle-bold text-bolt-elements-icon-error text-2xl" />;
            }
          }

          return undefined;
        }}
        position="bottom-right"
        pauseOnFocusLoss
      />
      {/* Conditionally render APIDebugPanel with error boundary */}
      {SHOW_RAW_API_RESPONSES && <SimpleDebugPanel />}
    </>
  );
}

const processSampledMessages = createSampler(
  (options: {
    messages: Message[];
    initialMessages: Message[];
    isLoading: boolean;
    parseMessages: (messages: Message[], isLoading: boolean) => void;
    storeMessageHistory: (messages: Message[]) => Promise<void>;
  }) => {
    const { messages, initialMessages, isLoading, parseMessages, storeMessageHistory } = options;
    parseMessages(messages, isLoading);

    if (messages.length > initialMessages.length) {
      storeMessageHistory(messages).catch((error) => toast.error(error.message));
    }
  },
  50,
);

interface ChatProps {
  initialMessages: Message[];
  storeMessageHistory: (messages: Message[]) => Promise<void>;
  importChat: (description: string, messages: Message[]) => Promise<void>;
  exportChat: () => void;
  description?: string;
}

export const ChatImpl = memo(
  ({ description, initialMessages, storeMessageHistory, importChat, exportChat }: ChatProps) => {
    useShortcuts();

    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const [chatStarted, setChatStarted] = useState(initialMessages.length > 0);
    const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
    const [imageDataList, setImageDataList] = useState<string[]>([]);
    const [searchParams, setSearchParams] = useSearchParams();
    const [fakeLoading, setFakeLoading] = useState(false);
    const files = useStore(workbenchStore.files);
    const actionAlert = useStore(workbenchStore.alert);
    const { contextOptimizationEnabled } = useSettings();
    const [apiStatus, setApiStatus] = useState<'checking' | 'ready' | 'error'>('checking');
    const [apiError, setApiError] = useState<string | null>(null);

    const { showChat } = useStore(chatStore);

    const [animationScope, animate] = useAnimate();

    // Check API health on component mount
    useEffect(() => {
      const checkApiHealth = async () => {
        try {
          // Initialize workbenchStore.debugMessages if it doesn't exist yet
          if (!workbenchStore.debugMessages.get()) {
            workbenchStore.debugMessages.set([]);
          }
          
          const response = await fetch('/api/health');
          
          // Save raw response text for debugging
          let rawResponseText = '';
          try {
            rawResponseText = await response.clone().text();
          } catch (textError) {
            rawResponseText = 'Failed to get response text';
          }
          
          let rawData: any = { status: 'unknown' };
          
          try {
            // Try to parse JSON response
            rawData = JSON.parse(rawResponseText);
          } catch (parseError) {
            // If not valid JSON, use the raw text
            rawData = { rawText: rawResponseText, status: 'parse_error' };
          }
          
          // Type the response explicitly with safe fallbacks
          const data: ApiHealthResponse = {
            status: rawData.status || 'unknown',
            ...rawData
          };
          
          // Log API health information to the console (only in development)
          if (process.env.NODE_ENV === 'development') {
            console.info('API Health Check:', data);
          }
          
          if (SHOW_RAW_API_RESPONSES) {
            // Safely add debug message with error handling
            try {
              workbenchStore.addDebugMessage({
                type: 'api-status',
                content: `API Status: ${JSON.stringify(data, null, 2)}`,
                timestamp: new Date().toISOString()
              });
            } catch (debugError) {
              console.error('Error adding debug message:', debugError);
            }
          }
          
          // Only log successful health checks
          if (data.status === 'ok') {
            logger.info('API health check:', data);
            setApiStatus('ready');
          } else {
            // Silent failure for error states but store information
            setApiStatus('error');
            setApiError(data.status);
          }
        } catch (err) {
          // Silent failure for connection errors but store more details
          const errorMessage = err instanceof Error ? err.message : 'Unknown connection error';
          setApiStatus('error');
          setApiError('connection_failed');
          
          if (SHOW_RAW_API_RESPONSES) {
            // Safely add error debug message
            try {
              workbenchStore.addDebugMessage({
                type: 'api-error',
                content: `API Connection Error: ${errorMessage}`,
                timestamp: new Date().toISOString()
              });
            } catch (debugError) {
              console.error('Error adding debug message:', debugError);
            }
          }
        }
      };
      
      // Wrap in try-catch to prevent component from crashing
      try {
        checkApiHealth();
      } catch (err) {
        console.error('Critical error in API health check:', err);
        setApiStatus('error');
        setApiError('critical_error');
      }
    }, []);

    const {
      messages,
      isLoading,
      input,
      handleInputChange,
      setInput,
      stop,
      append,
      setMessages,
      reload,
      error,
      data: chatData,
      setData,
    } = useStreamingChat({
      body: {
        files,
        contextOptimization: contextOptimizationEnabled,
      },
      sendExtraMessageFields: true,
      onError: (e) => {
        // Silent error logging - only log to debug
        logger.debug('Request issue occurred:', e.message);
        logStore.logError('Chat request issue', e, {
          component: 'Chat',
          action: 'request',
          error: e.message,
        });
        // No toast error displayed
      },
      onFinish: (message) => {
        setData(undefined);
        logger.debug('Finished streaming');
      },
      initialMessages,
      initialInput: Cookies.get(PROMPT_COOKIE_KEY) || '',
    });

    useEffect(() => {
      const prompt = searchParams.get('prompt');

      if (prompt) {
        setSearchParams({});
        runAnimation();
        append({
          role: 'user',
          content: prompt,
          id: crypto.randomUUID(),
        });
      }
    }, [searchParams]);

    const { parsedMessages, parseMessages } = useMessageParser();

    const TEXTAREA_MAX_HEIGHT = chatStarted ? 400 : 200;

    useEffect(() => {
      chatStore.setKey('started', initialMessages.length > 0);
    }, []);

    useEffect(() => {
      processSampledMessages({
        messages,
        initialMessages,
        isLoading,
        parseMessages,
        storeMessageHistory,
      });
    }, [messages, isLoading, parseMessages]);

    const scrollTextArea = () => {
      const textarea = textareaRef.current;

      if (textarea) {
        textarea.scrollTop = textarea.scrollHeight;
      }
    };

    const abort = () => {
      stop();
      chatStore.setKey('aborted', true);
      workbenchStore.abortAllActions();

      logStore.logProvider('Chat response aborted', {
        component: 'Chat',
        action: 'abort',
      });
    };

    useEffect(() => {
      const textarea = textareaRef.current;

      if (textarea) {
        textarea.style.height = 'auto';

        const scrollHeight = textarea.scrollHeight;

        textarea.style.height = `${Math.min(scrollHeight, TEXTAREA_MAX_HEIGHT)}px`;
        textarea.style.overflowY = scrollHeight > TEXTAREA_MAX_HEIGHT ? 'auto' : 'hidden';
      }
    }, [input, textareaRef]);

    const runAnimation = async () => {
      if (chatStarted) {
        return;
      }

      await Promise.all([
        animate('#examples', { opacity: 0, display: 'none' }, { duration: 0.1 }),
        animate('#intro', { opacity: 0, flex: 1 }, { duration: 0.2, ease: cubicEasingFn }),
      ]);

      chatStore.setKey('started', true);

      setChatStarted(true);
    };

    // Direct API chat function for first message
    const sendDirectApiChatMessage = async (message: string) => {
      try {
        logger.info('Sending direct API message:', message);
        setFakeLoading(true);
        
        // Store raw response for debugging display
        let rawResponseData = null;
        
        const response = await fetch(`${API_URL}/api/chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message,
            thread_id: undefined,  // First message doesn't have a thread ID
          }),
        });
        
        // Save raw response text for debugging
        let rawResponseText = '';
        try {
          rawResponseText = await response.clone().text();
        } catch (textError) {
          rawResponseText = 'Failed to get response text';
        }
        
        if (!response.ok) {
          // Display error in UI when SHOW_RAW_API_RESPONSES is true
          if (SHOW_RAW_API_RESPONSES) {
            const errorContent = `API Error (${response.status}): ${rawResponseText}`;
            
            // Add to debug messages
            try {
              workbenchStore.addDebugMessage({
                type: 'api-error',
                content: errorContent,
                timestamp: new Date().toISOString()
              });
            } catch (debugError) {
              console.error('Error adding debug message:', debugError);
            }
            
            setMessages([
              {
                id: crypto.randomUUID(),
                role: 'user',
                content: message,
              },
              {
                id: crypto.randomUUID(),
                role: 'assistant',
                content: `⚠️ **API Error Response:**\n\`\`\`json\n${errorContent}\n\`\`\``,
              },
            ]);
          }
          setFakeLoading(false);
          return null;
        }
        
        try {
          // Try to parse JSON response
          rawResponseData = JSON.parse(rawResponseText);
        } catch (parseError) {
          // If not valid JSON, use the raw text
          rawResponseData = { rawText: rawResponseText, success: false };
        }
        
        // Type the response explicitly with safe defaults
        const data: ApiChatResponse = {
          response: rawResponseData?.response || "No response received",
          thread_id: rawResponseData?.thread_id || crypto.randomUUID(),
          success: !!rawResponseData?.success,
          ...rawResponseData
        };
        
        // Add to debug messages regardless of success
        if (SHOW_RAW_API_RESPONSES) {
          try {
            workbenchStore.addDebugMessage({
              type: data.success ? 'api-response' : 'api-error',
              content: `API Chat Response: ${JSON.stringify(rawResponseData, null, 2)}`,
              timestamp: new Date().toISOString()
            });
          } catch (debugError) {
            console.error('Error adding debug message:', debugError);
          }
        }
        
        // Only log successful responses
        if (data.success) {
          logger.info('Received API response:', data);
        }
        
        // Format the response text for better display
        const formattedResponse = data.response
          ? data.response.replace(/\n\n/g, '\n').trim()
          : 'No response content';
        
        // Create debug response to show raw API data when enabled
        const debugResponse = SHOW_RAW_API_RESPONSES 
          ? `**Response from API:**\n\`\`\`json\n${JSON.stringify(rawResponseData, null, 2)}\n\`\`\`\n\n${formattedResponse}`
          : formattedResponse;
        
        // Update messages with user input and response
        setMessages([
          {
            id: data.thread_id || crypto.randomUUID(),
            role: 'user',
            content: message,
          },
          {
            id: `response-${Date.now()}`,
            role: 'assistant',
            content: debugResponse,
          },
        ]);
        
        // Silent success - no toast notification
        setFakeLoading(false);
        
        // Use the thread_id for future WebSocket connections
        return data.thread_id;
      } catch (err) {
        // Show error in UI when debugging is enabled
        if (SHOW_RAW_API_RESPONSES) {
          const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
          
          // Add to debug messages
          try {
            workbenchStore.addDebugMessage({
              type: 'api-error',
              content: `API Exception: ${errorMessage}`,
              timestamp: new Date().toISOString()
            });
          } catch (debugError) {
            console.error('Error adding debug message:', debugError);
          }
          
          setMessages([
            {
              id: crypto.randomUUID(),
              role: 'user',
              content: message,
            },
            {
              id: crypto.randomUUID(),
              role: 'assistant',
              content: `⚠️ **Exception Error:**\n\`\`\`\n${errorMessage}\n\`\`\``,
            },
          ]);
        }
        
        // Silent error handling
        setFakeLoading(false);
        return null;
      }
    };

    const sendMessage = async (_event: React.UIEvent, messageInput?: string) => {
      const messageContent = messageInput || input;

      if (!messageContent?.trim()) {
        return;
      }

      if (isLoading) {
        abort();
        return;
      }

      runAnimation();

      if (!chatStarted) {
        try {
          // For first message, use direct API call
          await sendDirectApiChatMessage(messageContent);
          
          setInput('');
          Cookies.remove(PROMPT_COOKIE_KEY);
          setUploadedFiles([]);
          setImageDataList([]);
          textareaRef.current?.blur();
          return;
        } catch (err) {
          // If API fails, fallback to the streaming chat
          logger.error('Failed to use direct API, falling back to streaming:', err);
          // Continue with normal flow below
        }
      }

      if (error != null) {
        setMessages(messages.slice(0, -1));
      }

      const modifiedFiles = workbenchStore.getModifiedFiles();

      chatStore.setKey('aborted', false);

      if (modifiedFiles !== undefined) {
        const userUpdateArtifact = filesToArtifacts(modifiedFiles, `${Date.now()}`);
        append({
          role: 'user',
          content: `${userUpdateArtifact}${messageContent}`,
          id: crypto.randomUUID(),
        });

        workbenchStore.resetAllFileModifications();
      } else {
        append({
          role: 'user',
          content: messageContent,
          id: crypto.randomUUID(),
        });
      }

      setInput('');
      Cookies.remove(PROMPT_COOKIE_KEY);

      setUploadedFiles([]);
      setImageDataList([]);

      textareaRef.current?.blur();
    };

    /**
     * Handles the change event for the textarea and updates the input state.
     * @param event - The change event from the textarea.
     */
    const onTextareaChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
      handleInputChange(event);
    };

    /**
     * Debounced function to cache the prompt in cookies.
     * Caches the trimmed value of the textarea input after a delay to optimize performance.
     */
    const debouncedCachePrompt = useCallback(
      debounce((event: React.ChangeEvent<HTMLTextAreaElement>) => {
        const trimmedValue = event.target.value.trim();
        Cookies.set(PROMPT_COOKIE_KEY, trimmedValue, { expires: 30 });
      }, 1000),
      [],
    );

    const [messageRef, scrollRef] = useSnapScroll();

    return (
      <BaseChat
        ref={animationScope}
        textareaRef={textareaRef}
        input={input}
        showChat={showChat}
        chatStarted={chatStarted}
        isStreaming={isLoading || fakeLoading}
        onStreamingChange={(streaming) => {
          streamingState.set(streaming);
        }}
        sendMessage={sendMessage}
        messageRef={messageRef}
        scrollRef={scrollRef}
        handleInputChange={(e) => {
          onTextareaChange(e);
          debouncedCachePrompt(e);
        }}
        handleStop={abort}
        description={description}
        importChat={importChat}
        exportChat={exportChat}
        messages={messages.map((message, i) => {
          if (message.role === 'user') {
            return message;
          }

          return {
            ...message,
            content: parsedMessages[i] || '',
          };
        })}
        uploadedFiles={uploadedFiles}
        setUploadedFiles={setUploadedFiles}
        imageDataList={imageDataList}
        setImageDataList={setImageDataList}
        actionAlert={actionAlert}
        clearAlert={() => workbenchStore.clearAlert()}
        data={chatData}
      />
    );
  },
);

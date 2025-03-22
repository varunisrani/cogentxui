import React from 'react';
import { useStore } from '@nanostores/react';
import { workbenchStore } from '~/lib/stores/workbench';
import { SHOW_RAW_API_RESPONSES } from '~/utils/constants';

const APIDebugPanel: React.FC = () => {
  // Add try-catch to prevent rendering errors
  try {
    const debugMessages = useStore(workbenchStore.debugMessages);
    
    // Safer null checking and conditions
    if (!SHOW_RAW_API_RESPONSES || !debugMessages || debugMessages.length === 0) {
      return null;
    }
    
    return (
      <div className="fixed bottom-4 right-4 z-50 max-w-md max-h-96 overflow-auto bg-black bg-opacity-90 text-white rounded-lg p-4 border border-gray-700 text-xs">
        <div className="flex justify-between items-center mb-2">
          <h3 className="font-bold">API Debug Information</h3>
          <button 
            onClick={() => workbenchStore.debugMessages.set([])}
            className="text-gray-400 hover:text-white"
          >
            Clear
          </button>
        </div>
        <div className="space-y-2">
          {debugMessages.map((message, index) => (
            <div key={index} className={`p-2 rounded ${message.type === 'api-error' ? 'bg-red-900 bg-opacity-40' : 'bg-gray-800'}`}>
              <div className="font-semibold text-xs mb-1">{message.type} - {new Date(message.timestamp).toLocaleTimeString()}</div>
              <pre className="whitespace-pre-wrap text-xs overflow-x-auto">{message.content}</pre>
            </div>
          ))}
        </div>
      </div>
    );
  } catch (error) {
    // Silent error handling prevents component from crashing the app
    console.error("Error rendering APIDebugPanel:", error);
    return null;
  }
};

export default APIDebugPanel; 
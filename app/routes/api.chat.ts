import type { ActionFunctionArgs } from "@remix-run/node";
import { json } from "@remix-run/node";

interface ChatRequestData {
  message: string;
  thread_id?: string;
  [key: string]: any;
}

/**
 * This is a proxy endpoint that forwards requests to the FastAPI backend
 */
export async function action({ request }: ActionFunctionArgs) {
  try {
    // Get the request data
    const requestData = await request.json() as ChatRequestData;
    
    // Extract message and thread_id
    const { message, thread_id } = requestData;

    if (!message) {
      return json({ error: "Message is required" }, { status: 400 });
    }
    
    // Forward the request to the FastAPI backend - hardcoded for now
    const apiUrl = 'http://localhost:8001';
    console.log(`Sending request to ${apiUrl}/api/chat`);
    
    try {
      const response = await fetch(`${apiUrl}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          thread_id,
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "Unknown error" }));
        console.error("Error from FastAPI:", errorData);
        return json(errorData, { status: response.status });
      }
      
      const data = await response.json();
      return json(data);
    } catch (fetchError) {
      console.error("Error connecting to FastAPI:", fetchError);
      
      // Return a more specific error
      return json({ 
        error: "Could not connect to the API server. Make sure the FastAPI server is running on port 8001.",
        details: fetchError instanceof Error ? fetchError.message : "Unknown error"
      }, { status: 503 });
    }
  } catch (error) {
    console.error('Error in chat API proxy:', error);
    return json({ error: "Internal server error" }, { status: 500 });
  }
}

import { json } from "@remix-run/node";
import type { LoaderFunctionArgs } from "@remix-run/node";

/**
 * This endpoint is not actually used directly but serves as a reminder of the WebSocket endpoint we're using.
 * The WebSocket connection is established directly from the client to:
 * ws://{host}/ws/chat/{threadId}
 * 
 * If we need to proxy WebSocket connections in the future, we can implement that here.
 */
export async function loader({ params }: LoaderFunctionArgs) {
  const { threadId } = params;
  
  return json({
    message: "WebSocket endpoint reminder",
    info: `The actual WebSocket connection should be made to ws://{host}/ws/chat/${threadId}`
  });
} 
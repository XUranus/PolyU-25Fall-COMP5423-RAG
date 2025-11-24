// src/components/ChatPanel.tsx
import React, { useState, useEffect } from 'react';

// Define the type for the raw message object received from the API
interface RawMessage {
  id: string; // Using UUID string
  sender: 'user' | 'bot';
  content: string; // API returns 'content'
  timestamp?: string;
  thinkingProcess?: string; // Stored as JSON string in DB/API
  retrieved_docs?: string; // Stored as JSON string in DB/API
}

// Define the type for the message after processing (parsing JSON strings)
interface ProcessedMessage {
  id: string;
  sender: 'user' | 'bot';
  text: string; // Use 'text' for consistency in rendering
  timestamp?: string;
  thinkingProcess?: string[]; // Parsed array
  retrieved_docs?: [string, number][]; // Parsed array of tuples
}

interface ChatPanelProps {
  currentChatId: string | null; // Receive the ID of the chat to display
}

const ChatPanel: React.FC<ChatPanelProps> = ({ currentChatId }) => {
  const [messages, setMessages] = useState<ProcessedMessage[]>([]);

  // Fetch messages whenever currentChatId changes
  useEffect(() => {
    const fetchMessages = async () => {
      if (!currentChatId) {
        // If no chat is selected, clear messages
        setMessages([]);
        return;
      }

      try {
        console.log(`Fetching messages for chat ID: ${currentChatId}`);
        const response = await fetch(`http://127.0.0.1:5001/api/chat/${currentChatId}/messages`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const rawData: RawMessage[] = await response.json();

        // Process the raw data: parse JSON strings and map to ProcessedMessage
        const processedMessages: ProcessedMessage[] = rawData.map(rawMsg => ({
          id: rawMsg.id,
          sender: rawMsg.sender,
          text: rawMsg.content, // Map 'content' from API to 'text' for display
          timestamp: rawMsg.timestamp,
          thinkingProcess: rawMsg.thinkingProcess ? JSON.parse(rawMsg.thinkingProcess) as string[] : undefined,
          retrieved_docs: rawMsg.retrieved_docs ? JSON.parse(rawMsg.retrieved_docs) as [string, number][] : undefined,
        }));

        setMessages(processedMessages);
      } catch (err) {
        console.error(`Failed to fetch messages for chat ${currentChatId}:`, err);
        setMessages([{ id: 'error', sender: 'bot', text: 'Error loading messages for this chat.' }]);
      }
    };

    fetchMessages();
  }, [currentChatId]); // Dependency array includes currentChatId

  if (!currentChatId) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        <p>Select an existing chat or create a new one to start messaging.</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {messages.map((msg) => (
        <div
          key={msg.id} // Using the UUID string as the key
          className={`flex ${
            msg.sender === 'user' ? 'justify-end' : 'justify-start'
          }`}
        >
          <div
            className={`max-w-[80%] p-3 rounded-lg ${
              msg.sender === 'user'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-800 text-white'
            }`}
          >
            {msg.sender === 'bot' && msg.thinkingProcess && (
              <div className="mb-2 p-2 bg-gray-700 rounded text-sm">
                <strong>View Thinking Process ({msg.thinkingProcess.length} steps)</strong>
                <div className="mt-1 space-y-1">
                  {msg.thinkingProcess.map((step, index) => (
                    <div key={index} className="flex items-start">
                      <span className="mr-1">↳</span>
                      <span>{step}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            <p>{msg.text}</p>
            {msg.retrieved_docs && (
              <details className="mt-1 text-xs text-gray-400">
                <summary>Retrieved Docs</summary>
                <ul>
                  {msg.retrieved_docs.map(([id, score], idx) => (
                    <li key={idx}>{id}: {score.toFixed(2)}</li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        </div>
      ))}
    </div>
  );
};

export default ChatPanel;
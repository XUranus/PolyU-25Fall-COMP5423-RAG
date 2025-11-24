// src/components/ChatPanel.tsx
import React, { useState, useEffect, useRef } from 'react';


interface MessageHttpResponse {
  id: string;
  sender: string,
  content: string;
  thinking_process?: string;
  retrieved_docs? : string;
  timpstamp: string;
}

interface Message {
  id: string;
  sender: 'user' | 'bot';
  content: string;
  thinkingProcess?: string[]; // Optional for bot messages
  retrievedDocs?: [string, number][]; // Optional for bot messages
}

function convertHttpResponseToMessage(httpResponse: MessageHttpResponse): Message {
  // Basic mapping
  const convertedMessage: Message = {
    id: httpResponse.id,
    sender: httpResponse.sender as 'user' | 'bot', // Type assertion after confirming source
    content: httpResponse.content,
    // timestamp is present in the response but not in the target Message interface
    // Add optional fields only if they exist in the response
  };

  // Parse thinking_process if it exists
  if (httpResponse.thinking_process) {
    try {
      // Attempt to parse the JSON string into an array of strings
      const parsedThinkingProcess: unknown = JSON.parse(httpResponse.thinking_process);
      // Type guard to ensure it's an array of strings
      if (Array.isArray(parsedThinkingProcess) && parsedThinkingProcess.every(item => typeof item === 'string')) {
        convertedMessage.thinkingProcess = parsedThinkingProcess as string[];
      } else {
        console.warn(`Parsed thinking_process is not an array of strings for message ${httpResponse.id}. Got:`, parsedThinkingProcess);
        // Optionally, you could set it to an empty array or skip the field if parsing fails strictly
        // convertedMessage.thinkingProcess = [];
      }
    } catch (error) {
      console.error(`Failed to parse thinking_process JSON for message ${httpResponse.id}:`, error);
      console.error(`Raw thinking_process string was:`, httpResponse.thinking_process);
      // Optionally, add an error message to the thinkingProcess field
      // convertedMessage.thinkingProcess = [`Error parsing thinking process: ${error.message}`];
    }
  }

  // Parse retrieved_docs if it exists
  if (httpResponse.retrieved_docs) {
    try {
      // Attempt to parse the JSON string into an array of [string, number] tuples
      const parsedRetrievedDocs: unknown = JSON.parse(httpResponse.retrieved_docs);
      // Type guard to ensure it's an array of [string, number] tuples
      if (Array.isArray(parsedRetrievedDocs) &&
          parsedRetrievedDocs.every(item => Array.isArray(item) && item.length === 2 && typeof item[0] === 'string' && typeof item[1] === 'number')) {
        convertedMessage.retrievedDocs = parsedRetrievedDocs as [string, number][];
      } else {
        console.warn(`Parsed retrieved_docs is not an array of [string, number] tuples for message ${httpResponse.id}. Got:`, parsedRetrievedDocs);
        // Optionally, set it to an empty array or skip the field
        // convertedMessage.retrievedDocs = [];
      }
    } catch (error) {
      console.error(`Failed to parse retrieved_docs JSON for message ${httpResponse.id}:`, error);
      console.error(`Raw retrieved_docs string was:`, httpResponse.retrieved_docs);
      // Optionally, add an error indicator
      // convertedMessage.retrievedDocs = [["Error", -1]]; // Or similar
    }
  }

  return convertedMessage;
}

interface ChatPanelProps {
  currentChatId: string | null; // Receive the active chat ID from App
}

const ChatPanel: React.FC<ChatPanelProps> = ({ currentChatId }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false); // To show a loading state while waiting for bot response
  const messagesEndRef = useRef<null | HTMLDivElement>(null); // For auto-scrolling

  // Fetch messages when the currentChatId changes
  useEffect(() => {
    if (currentChatId !== null) {
        const fetchMessages = async () => {
            try {
                const response = await fetch(`http://127.0.0.1:5001/api/chat/${currentChatId}/messages`);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                let all_messages : MessageHttpResponse[] = await response.json()
                const data: Message[] = all_messages.map(msg => convertHttpResponseToMessage(msg))
                console.log('raw ', all_messages)
                console.log('converted ', data)

                setMessages(data);
            } catch (error) {
                console.error("Failed to fetch messages:", error);
                // Set a default message or handle error state
                setMessages([{ id: "FAKEID-MESSAGE-FAILED", sender: 'bot', content: "Error loading chat history. Please try again." }]);
            }
        };
        fetchMessages();
    } else {
        setMessages([]); // Clear messages if no chat is selected
    }
  }, [currentChatId]);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    if (!inputText.trim() || !currentChatId || isLoading) return;

    const userMessage: Message = {
      id: "FAKEID-TEMP-MESSAGE", // Temporary ID, will be replaced by server ID later if needed
      sender: 'user',
      content: inputText,
    };

    // Optimistically add user message to UI
    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      const response = await fetch(`http://127.0.0.1:5001/api/chat/${currentChatId}/messages`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: inputText }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const botResponse: {
        id: string;
        sender: 'bot';
        content: string;
        thinking_process: string[];
        retrieved_docs: [string, number][];
      } = await response.json();

      // Add the bot's response to the messages
      setMessages(prev => [
        ...prev,
        {
          id: botResponse.id,
          sender: botResponse.sender,
          content: botResponse.content,
          thinkingProcess: botResponse.thinking_process,
          retrievedDocs: botResponse.retrieved_docs,
        }
      ]);
    } catch (error) {
      console.error("Failed to send message:", error);
      // Add an error message from the bot
      setMessages(prev => [...prev, { id: "FAKEID-MESSAGE-FAILED", sender: 'bot', content: "Sorry, an error occurred while processing your message." }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault(); // Prevents adding a new line in the textarea
      handleSend();
    }
  };

  return (
    <div className="flex-1 flex flex-col">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg) => (
          <div
            key={msg.id}
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
              <p>{msg.content}</p>

              {msg.sender === 'bot' && msg.retrievedDocs && (
                  <details className="mt-2 text-xs text-gray-400">
                      <summary>Retrieved Docs</summary>
                      <ul className="list-disc list-inside">
                          {msg.retrievedDocs.map(([docId, score], idx) => (
                              <li key={idx}>{docId} (Score: {score.toFixed(2)})</li>
                          ))}
                      </ul>
                  </details>
              )}
            </div>
          </div>
        ))}
        {isLoading && (
            <div className="flex justify-start">
                <div className="bg-gray-800 text-white p-3 rounded-lg max-w-[80%]">
                    <p>Thinking...</p>
                </div>
            </div>
        )}
        <div ref={messagesEndRef} />
      </div>
    </div>
  );
};

export default ChatPanel;
// src/components/ChatPanel.tsx
import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

import LoadingButton from './LoadingButton';
import ThinkingGlow from './ThinkingGlow';

const MarkdownRenderer: React.FC<{ markdownContent: string }> = ({ markdownContent }) => {
  return (
    <div className="prose prose-lg max-w-none dark:prose-invert">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>
        {markdownContent}
      </ReactMarkdown>
    </div>
  );
};

interface MessageHttpResponse {
  id: string;
  sender: string,
  content: string;
  thinking_process?: string;
  timestamp: string;
}

interface RetrievedDoc {
  docId: string;
  text: string;
  score: number;
}

interface ThinkingProcessStep {
  step: number,
  description: string,
  type : string,
  retrieved_docs? : RetrievedDoc[]
}

interface Message {
  id: string;
  sender: 'user' | 'bot';
  content: string;
  thinkingProcess?: ThinkingProcessStep[]; // Optional for bot messages
  timestamp: string;
}

function convertHttpResponseToMessage(httpResponse: MessageHttpResponse): Message {
  // Basic mapping
  const convertedMessage: Message = {
    id: httpResponse.id,
    sender: httpResponse.sender as 'user' | 'bot', // Type assertion after confirming source
    content: httpResponse.content,
    timestamp: httpResponse.timestamp,
  };

  // Parse thinking_process if it exists
  if (httpResponse.thinking_process) {
    try {
      // Attempt to parse the JSON string into an array of strings
      const parsedThinkingProcess: unknown = JSON.parse(httpResponse.thinking_process);
      // Type guard to ensure it's an array of strings
      if (Array.isArray(parsedThinkingProcess)) {
        convertedMessage.thinkingProcess = parsedThinkingProcess as ThinkingProcessStep[];
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
  return convertedMessage;
}


interface ChatPanelProps {
  currentChatId: string | null; // Receive the active chat ID from App
  currentModel : string
}

const ChatPanel: React.FC<ChatPanelProps> = ({ currentChatId, currentModel }) => {
  const [hoveredMessageId, setHoveredMessageId] = useState<string | null>(null); // Track which chat is hovered
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false); // To show a loading state while waiting for bot response
  const messagesEndRef = useRef<null | HTMLDivElement>(null); // For auto-scrolling


  // delete a message
  const handleDeleteMessageClick = async (messageId: string, e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent triggering onSelectChat if delete button is inside the list item button
    try {
        const response = await fetch(`http://127.0.0.1:5000/api/message/${messageId}`, {
            method: 'DELETE',
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        setMessages(messages.filter(msg => msg.id !== messageId))
    } catch (error) {
        console.error("Failed to delete chat:", error);
        alert("Failed to delete message. Please try again.");
    }
  };


  // Fetch messages when the currentChatId changes
  useEffect(() => {
    if (currentChatId !== null) {
        const fetchMessages = async () => {
            try {
                const response = await fetch(`http://127.0.0.1:5000/api/chat/${currentChatId}/messages`);
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
                setMessages([{
                  id: "FAKEID-MESSAGE-FAILED",
                  sender: 'bot',
                  timestamp : new Date().toLocaleString('sv-SE', {timeZone: 'Asia/Shanghai'}),
                  content: "Error loading chat history. Please try again." 
                }]);
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
    console.log('handleSend ', inputText)
    if (!inputText.trim() || !currentChatId || isLoading) return;

    const userMessage: Message = {
      id: "FAKEID-TEMP-MESSAGE", // Temporary ID, will be replaced by server ID later if needed
      sender: 'user',
      content: inputText,
      timestamp: new Date().toLocaleString('sv-SE', {timeZone: 'Asia/Shanghai'})
    };

    // Optimistically add user message to UI
    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      const response = await fetch(`http://127.0.0.1:5000/api/chat/${currentChatId}/messages`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: inputText, model_name : currentModel }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const botResponse: {
        id: string;
        sender: 'bot';
        content: string;
        timestamp: string;
        thinking_process: ThinkingProcessStep[];
      } = await response.json();

      // Add the bot's response to the messages
      setMessages((prev : Message[]) => [
        ...prev,
        {
          id: botResponse.id,
          sender: botResponse.sender,
          content: botResponse.content,
          thinkingProcess: botResponse.thinking_process,
          timestamp: new Date().toLocaleString('sv-SE', {timeZone: 'Asia/Shanghai'}),
        }
      ]);
    } catch (error) {
      console.error("Failed to send message:", error);
      // Add an error message from the bot
      setMessages((prev : Message[]) => [...prev, { 
          id: "FAKEID-MESSAGE-FAILED",
          sender: 'bot',
          content: "Sorry, an error occurred while processing your message.",
          timestamp: new Date().toLocaleString('sv-SE', {timeZone: 'Asia/Shanghai'})
      }]);
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

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputText(e.target.value); // Update the inputText state with the current value of the textarea
  };

  return (
    <div className="flex-1 flex flex-col h-full">
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map((msg) => (
            <div
                key={msg.id}
                onMouseEnter={() => setHoveredMessageId(msg.id)} // Set hovered chat ID
                onMouseLeave={() => setHoveredMessageId(null)}  // Clear hovered chat ID
                className={`flex ${
                msg.sender === 'user' ? 'justify-end' : 'justify-start'
                }`}
            >
                <div
                className={`max-w-[80%] p-3 rounded-lg ${
                    msg.sender === 'user'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-800 text-white'
                }`}>

                {msg.sender === 'bot' && msg.thinkingProcess && (
                    <div className="mb-2 p-2 bg-gray-700 rounded text-sm">
                      <details className="mt-2 p-2 text-gray-400" open>
                          <summary><strong>View Thinking Process ({msg.thinkingProcess.length} steps)</strong></summary>
                          <div className="mt-1 space-y-1">
                          {msg.thinkingProcess.map((thinkingStep : ThinkingProcessStep) => (
                            <div key={thinkingStep.step} className="flex items-start">
                                <span className="mr-1"></span>
                                <span><MarkdownRenderer markdownContent={thinkingStep.description} /></span>
                            </div>
                          ))}
                      </div>
                      </details>
                    </div>
                )}


                { /* Message Content */ }
                <MarkdownRenderer markdownContent={msg.content} />
                



              <hr className="my-4 border-t border-gray-300" />
              <div className="flex items-center justify-end">
                <span className="text-xs text-gray-200 whitespace-nowrap mr-2">{msg.timestamp}</span>
                
                {/* Fixed-size container for delete button — always reserves space, just hides content */}
                <div className="w-6 h-6 flex items-center justify-center">
                  {hoveredMessageId === msg.id && (
                    <button
                      onClick={(e) => handleDeleteMessageClick(msg.id, e)}
                      className="p-1 rounded hover:bg-red-600 transition-colors"
                      aria-label="Delete message"
                    >
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="h-4 w-4 text-gray-400 hover:text-white"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M6 18L18 6M6 6l12 12"
                        />
                      </svg>
                    </button>
                  )}
                </div>
              </div>

                              

              </div>
            </div>
            ))}
            {isLoading && (
                <div className="flex justify-start">
                    <div className="bg-gray-800 text-white p-3 rounded-lg max-w-[80%]">
                        <p>Thinking...</p>
                        <ThinkingGlow isThinking={true} text={'x'.repeat(50)}/>
                    </div>
                </div>
            )}
            <div ref={messagesEndRef} />
        </div>

        {/* Fixed footer with input area */}
        <footer className="bg-gray-800 p-4 border-t border-gray-700">
            <div className="flex items-center space-x-2">
            <textarea
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                value={inputText}
                placeholder="Message Group42 AI..."
                className="flex-1 p-2 rounded bg-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            />

            <LoadingButton
              loading={isLoading}
              onClick={handleSend}
              noinput = {!inputText.trim() || !currentChatId}
              disabled={isLoading || !inputText.trim() || !currentChatId}>
                Submit
            </LoadingButton>
            </div>
      </footer>
    </div>
  );
};

export default ChatPanel;
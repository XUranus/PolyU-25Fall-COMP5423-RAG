// src/components/ChatPanel.tsx
import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';

import LoadingButton from './LoadingButton';
import ThinkingPanel from './ThinkingPanel'
import {
  MessageHttpResponse, ThinkingGlow, Message,
  convertHttpResponseToMessage, ThinkingProcessStep, MarkdownRenderer }
  from './ThinkingPanel'
import { API_PREFIX } from '../config'

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
  const navigate = useNavigate()

  // delete a message
  const handleDeleteMessageClick = async (messageId: string, e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent triggering onSelectChat if delete button is inside the list item button
    try {
        const response = await fetch(`${API_PREFIX}/api/message/${messageId}`, {
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
                const response = await fetch(`${API_PREFIX}/api/chat/${currentChatId}/messages`);
                if (!response.ok) {
                    window.location.href = "/";
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
                  id: "FAKEID-MESSAGE-FAILED" + Date.now(),
                  sender: 'bot',
                  timestamp : new Date().toLocaleString('sv-SE', {timeZone: 'Asia/Shanghai'}),
                  content: "❌ Whoops! There seems to be some network issue, please try again later." 
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

    let temp_user_message_id = "FAKEID-TEMP-MESSAGE" + Date.now()
    const timestamp = new Date().toLocaleString('sv-SE', {timeZone: 'Asia/Shanghai'})

    const userMessage: Message = {
      id: temp_user_message_id, // Temporary ID, will be replaced by server ID
      sender: 'user',
      content: inputText,
      timestamp,
    };

    // Optimistically add user message to UI
    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      const response = await fetch(`${API_PREFIX}/api/chat/${currentChatId}/messages`, {
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
        user_message_id : string,
        sender: 'bot';
        content: string;
        timestamp: string;
        thinking_process: ThinkingProcessStep[];
      } = await response.json();

      const lastMessage : Message = {
        id: botResponse.user_message_id,
        sender: 'user',
        content: inputText,
        timestamp,
      };
      
      let newMessages = messages.filter(message => message.id != temp_user_message_id)
      newMessages = [
        ...newMessages,
        lastMessage,
        {
          id: botResponse.id,
          sender: botResponse.sender,
          content: botResponse.content,
          thinkingProcess: botResponse.thinking_process,
          timestamp: new Date().toLocaleString('sv-SE', {timeZone: 'Asia/Shanghai'}),
        }
      ]
      // Add the bot's response to the messages
      setMessages(newMessages);
    } catch (error) {
      console.error("Failed to send message:", error);
      // Add an error message from the bot
      setMessages((prev : Message[]) => [...prev, { 
          id: "FAKEID-MESSAGE-FAILED" + Date.now(),
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
                <div>
                  <span className="text-xs text-gray-500 whitespace-nowrap mr-2"><i>{msg.timestamp}</i></span>
                  <div className={`max-w-[100%] p-3 rounded-lg ${
                        msg.sender === 'user'
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-800 text-white'
                    }`}>

                    {msg.sender === 'bot' && msg.thinkingProcess && (
                      <ThinkingPanel steps={msg.thinkingProcess}/>
                    )}

                    { /* Message Content */ }
                    <MarkdownRenderer markdownContent={msg.content} />
                  


                    <div className="flex items-center justify-end">
                      
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
            </div>
            ))}
            {isLoading && (
                <div className="flex justify-start">
                    <div className="mb-2 p-2 bg-gray-700 rounded text-sm">
                      <div className='cursor-pointer hover:bg-gray-600 p-1 rounded bg-gray-600'>
                        <strong><p>💭 Thinking... </p></strong>
                        <ThinkingGlow isThinking={true} text={'x'.repeat(50)}/>
                      </div>
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
              onClick={()=>{handleSend()}}
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
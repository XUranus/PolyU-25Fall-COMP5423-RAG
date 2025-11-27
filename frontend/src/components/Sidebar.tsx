// src/components/Sidebar.tsx
import React, { useState } from 'react';
import { API_PREFIX } from '../config'

interface ChatSession {
  id: string;
  title: string;
  updated_at: string; // ISO string format
}

interface SidebarProps {
  chats: ChatSession[];
  currentChatId: string | null;
  onNewChat: () => void;
  onSelectChat: (chatId: string) => void;
  onDeleteChat: (chatId: string) => void; // Add onDeleteChat prop
}

const Sidebar: React.FC<SidebarProps> = ({ chats, currentChatId, onNewChat, onSelectChat, onDeleteChat }) => {
  const [hoveredChatId, setHoveredChatId] = useState<string | null>(null); // Track which chat is hovered

  const handleDeleteClick = async (chatId: string, e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent triggering onSelectChat if delete button is inside the list item button
    if (window.confirm(`Are you sure you want to delete the chat "${chats.find(c => c.id === chatId)?.title || 'Untitled'}"?`)) {
        try {
            const response = await fetch(API_PREFIX + `/api/chat/${chatId}`, {
                method: 'DELETE',
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            // If deletion is successful, call the parent handler to update the state
            onDeleteChat(chatId);
        } catch (error) {
            console.error("Failed to delete chat:", error);
            alert("Failed to delete chat. Please try again.");
        }
    }
  };

  return (
    <div className="h-full flex flex-col bg-gray-800">
      {/* New Chat Button */}
      <div className="p-4 border-b border-gray-700">
        <button
          onClick={onNewChat}
          className="w-full flex items-center space-x-2 p-2 rounded hover:bg-gray-700 transition-colors"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
          </svg>
          <span>New Chat</span>
        </button>
      </div>

      {/* History Section */}
      <div className="flex-1 overflow-y-auto p-4"> {/* Make the history scrollable */}
        <h2 className="text-xs uppercase text-gray-400 mb-2">HISTORY</h2>
        <ul className="space-y-1">
          {chats.map((chat : ChatSession) => (
            <li key={chat.id} className="relative"> {/* Add relative positioning for delete button */}
              <div
                onClick={() => onSelectChat(chat.id)}
                className={`w-full text-left p-2 rounded transition-colors flex justify-between items-center ${
                  currentChatId === chat.id ? 'bg-gray-600' : 'hover:bg-gray-700'
                }`}
                onMouseEnter={() => setHoveredChatId(chat.id)} // Set hovered chat ID
                onMouseLeave={() => setHoveredChatId(null)}  // Clear hovered chat ID
              >
                <div className="flex items-center">
                  {chat.title}
                </div>
                {/* Delete Button - Show only on hover */}
                {hoveredChatId === chat.id && (
                  <button
                    onClick={(e) => handleDeleteClick(chat.id, e)} // Call delete handler
                    className="p-1 rounded hover:bg-red-600" // Add hover effect for delete button
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-gray-400 hover:text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                )}
              </div>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default Sidebar;
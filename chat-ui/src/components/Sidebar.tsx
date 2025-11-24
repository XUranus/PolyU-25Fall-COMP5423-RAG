// src/components/Sidebar.tsx
import React, { useState } from 'react';

interface ChatHistoryItem {
  id: number;
  title: string;
}

const Sidebar: React.FC = () => {
  const [chats, setChats] = useState<ChatHistoryItem[]>([
    { id: 1, title: "The second plac..." },
    { id: 2, title: "Who won the race?" },
    { id: 3, title: "What team was he on?" },
  ]);

  const handleNewChat = () => {
    const newId = chats.length + 1;
    setChats([...chats, { id: newId, title: `New Chat ${newId}` }]);
  };

  return (
    <div className="h-full flex flex-col bg-gray-800">
      {/* New Chat Button */}
      <div className="p-4 border-b border-gray-700">
        <button
          onClick={handleNewChat}
          className="w-full flex items-center space-x-2 p-2 rounded hover:bg-gray-700 transition-colors"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
          </svg>
          <span>New Chat</span>
        </button>
      </div>

      {/* History Section */}
      <div className="p-4">
        <h2 className="text-xs uppercase text-gray-400 mb-2">HISTORY</h2>
        <ul className="space-y-1">
          {chats.map((chat) => (
            <li key={chat.id}>
              <button className="w-full text-left p-2 rounded hover:bg-gray-700 transition-colors">
                <svg xmlns="http://www.w3.org/2000/svg" className="inline h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h8v6H8v-6z" />
                </svg>
                {chat.title}
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default Sidebar;
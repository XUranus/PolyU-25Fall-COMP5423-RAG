// src/App.tsx (Corrected - Input area logic stays in ChatPanel)
import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatPanel from './components/ChatPanel'; // Import ChatPanel

// Define types for our data structures
interface ChatSession {
  id: string;
  title: string;
  updated_at: string; // ISO string format
}

function App() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null); // Track the active chat
  const [availableChats, setAvailableChats] = useState<ChatSession[]>([]); // Store chat history list

  // Fetch chat history list on app load or when needed
  useEffect(() => {
    const fetchChatHistory = async () => {
        try {
            const response = await fetch('http://127.0.0.1:5001/api/chats/list');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data: ChatSession[] = await response.json();
            setAvailableChats(data);
            // Optionally, set the first chat as active if none is selected and list is not empty
            if (data.length > 0 && currentChatId === null) {
                setCurrentChatId(data[0].id);
            }
        } catch (error) {
            console.error("Failed to fetch chat history:", error);
            setAvailableChats([]);
        }
    };
    fetchChatHistory();
  }, [currentChatId]); // Add currentChatId to dependency if fetching history affects active chat logic

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  // Handler to be passed to Sidebar to create a new chat
  const handleNewChat = async () => {
    try {
        const response = await fetch('http://127.0.0.1:5001/api/chats/new', {
            method: 'POST',
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const newChatData: { id: string; title: string } = await response.json();
        // Add the new chat to the list and set it as active
        setAvailableChats(prev => [{ id: newChatData.id, title: newChatData.title, updated_at: new Date().toISOString() }, ...prev]);
        setCurrentChatId(newChatData.id);
    } catch (error) {
        console.error("Failed to create new chat:", error);
    }
  };

  // Handler to be passed to Sidebar to select a chat
  const handleSelectChat = (chatId: string) => {
    setCurrentChatId(chatId);
  };

  // New handler for deleting a chat
  const handleDeleteChat = (chatId: string) => {
    // Remove the chat from the local state
    setAvailableChats(prev => prev.filter(chat => chat.id !== chatId));
    // If the deleted chat was the current one, clear the currentChatId
    if (currentChatId === chatId) {
        setCurrentChatId(null);
        // Optionally, select the next available chat or clear the chat panel
        // For example, select the first chat if available:
        // if (availableChats.length > 1) {
        //   const remainingChats = availableChats.filter(chat => chat.id !== chatId);
        //   setCurrentChatId(remainingChats[0].id);
        // } else {
        //   setCurrentChatId(null); // Or keep it null if no chats remain
        // }
    }
};

  return (
    <div className="flex h-screen bg-gray-900 text-white">
      {/* Sidebar */}
      <div
        className={`${
          isSidebarOpen ? 'w-80' : 'w-0'
        } transition-all duration-300 ease-in-out overflow-hidden md:w-80 bg-gray-800`}
      >
        <Sidebar
          chats={availableChats}
          currentChatId={currentChatId}
          onNewChat={handleNewChat}
          onSelectChat={handleSelectChat}
          onDeleteChat={handleDeleteChat}
        />
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col h-full"> 
        {/* Header */}
        <header className="bg-gray-800 p-4 flex justify-between items-center border-b border-gray-700">
          <button
            onClick={toggleSidebar}
            className="md:hidden p-2 rounded hover:bg-gray-700"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          <h1 className="text-xl font-bold">Group4 AI</h1>
          <div className="hidden md:block">
            <select className="bg-gray-700 text-white px-3 py-1 rounded">
              <option>Multi-hop RAG</option>
              <option>Basic RAG</option>
              <option>Hybrid Retrieval</option>
            </select>
          </div>
        </header>

        {/* Chat Panel */}
        <main className="flex-1 overflow-hidden">
          {/* Pass currentChatId to ChatPanel */}
          <ChatPanel currentChatId={currentChatId} />

          {/* The input area is now handled inside ChatPanel */}
        </main>
      </div>
    </div>
  );
}

export default App;
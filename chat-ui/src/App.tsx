// src/App.tsx
import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatPanel from './components/ChatPanel';

function App() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  // State to track the currently active chat session ID
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  // Handler called by Sidebar when a new chat is created
  const handleNewChatCreated = (newChatId: string) => {
    console.log("App notified of new chat ID:", newChatId);
    setCurrentChatId(newChatId); // Switch to the newly created chat
  };

  // Handler called by Sidebar when an existing chat is selected
  const handleChatSelected = (chatId: string) => {
    console.log("App notified of selected chat ID:", chatId);
    setCurrentChatId(chatId); // Switch to the selected chat
  };

  // Optional: Fetch messages for the current chat when currentChatId changes
  // This could be handled inside ChatPanel too, depending on your preference
  // useEffect(() => {
  //   if (currentChatId) {
  //     // Fetch messages for currentChatId here if needed in App
  //     // Or pass currentChatId down to ChatPanel and let it manage its own fetching
  //   }
  // }, [currentChatId]);

  return (
    <div className="flex h-screen bg-gray-900 text-white">
      {/* Sidebar */}
      <div
        className={`${
          isSidebarOpen ? 'w-80' : 'w-0'
        } transition-all duration-300 ease-in-out overflow-hidden md:w-80`}
      >
        {/* Pass handler functions and currentChatId to Sidebar */}
        <Sidebar
          onNewChatCreated={handleNewChatCreated}
          onChatSelected={handleChatSelected}
          currentChatId={currentChatId}
        />
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="bg-gray-800 p-4 flex justify-between items-center">
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
        <main className="flex-1 overflow-y-auto p-4">
          {/* Pass the currentChatId to ChatPanel so it knows which messages to load */}
          <ChatPanel currentChatId={currentChatId} />
        </main>

        {/* Input Area */}
        <footer className="bg-gray-800 p-4">
          <div className="flex items-center space-x-2">
            <input
              type="text"
              placeholder={currentChatId ? "Message Group4 AI..." : "Select or create a chat to start"} // Disable placeholder hint if no chat is active
              className="flex-1 p-2 rounded bg-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={!currentChatId} // Disable input if no chat is active
            />
            <button 
              className="p-2 bg-blue-600 rounded hover:bg-blue-700"
              disabled={!currentChatId} >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 9M12 19l-9 9m9-9v-6m6 6a3 3 0 100-6 3 3 0 000 6z" />
              </svg>
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default App;
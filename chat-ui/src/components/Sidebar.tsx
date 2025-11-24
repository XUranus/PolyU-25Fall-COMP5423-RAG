// src/components/Sidebar.tsx
import React, { useState, useEffect, useRef } from 'react';

interface ChatHistoryItem {
  id: string; // Now using UUID string
  title: string;
  updated_at?: string;
}

interface SidebarProps {
  onNewChatCreated: (newChatId: string) => void;
  onChatSelected: (chatId: string) => void;
  currentChatId: string | null;
}

const Sidebar: React.FC<SidebarProps> = ({ onNewChatCreated, onChatSelected, currentChatId }) => {
  const [chats, setChats] = useState<ChatHistoryItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // State to manage which chat's menu is open and its position
  const [openMenuId, setOpenMenuId] = useState<string | null>(null);
  const [menuPosition, setMenuPosition] = useState({ top: 0, left: 0 });

  // Ref for the menu to handle clicks outside
  const menuRef = useRef<HTMLDivElement>(null);

  // --- Fetch Chat History on Component Mount ---
  useEffect(() => {
    const fetchChatHistory = async () => {
      try {
        const response = await fetch('http://127.0.0.1:5001/api/chats/list');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data: ChatHistoryItem[] = await response.json();
        setChats(data);
      } catch (err) {
        console.error("Failed to fetch chat history:", err);
        setError("Failed to load chat history. Please check the backend server.");
        setChats([]);
      } finally {
        setLoading(false);
      }
    };

    fetchChatHistory();
  }, []);

  // --- Handle Click Outside Menu ---
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setOpenMenuId(null);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // --- Handle New Chat Button Click ---
  const handleNewChat = async () => {
    try {
      const response = await fetch('http://127.0.0.1:5001/api/chats/new', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const newChatData: { id: string; title: string } = await response.json();
      console.log("New chat created:", newChatData);

      setChats(prevChats => [
        { id: newChatData.id, title: newChatData.title },
        ...prevChats
      ]);

      onNewChatCreated(newChatData.id);
    } catch (err) {
      console.error("Failed to create new chat:", err);
      alert("Failed to create a new chat. Please try again.");
    }
  };

  // --- Handle Chat Click ---
  const handleChatClick = (chatId: string) => {
    onChatSelected(chatId);
    setOpenMenuId(null); // Close menu if opened on a different item
  };

  // --- Handle Menu Button Click ---
  const handleMenuClick = (e: React.MouseEvent, chatId: string) => {
    e.stopPropagation(); // Prevent triggering the chat click handler
    if (openMenuId === chatId) {
      setOpenMenuId(null); // Close if clicking the same menu again
    } else {
      // Calculate position relative to the button clicked
      const rect = (e.target as HTMLElement).getBoundingClientRect();
      setMenuPosition({ top: rect.bottom, left: rect.left });
      setOpenMenuId(chatId);
    }
  };

  // --- Handle Delete Action ---
  const handleDelete = async (chatId: string) => {
    if (!window.confirm(`Are you sure you want to delete the chat "${chats.find(c => c.id === chatId)?.title}"?`)) {
      return; // Exit if user cancels
    }

    try {
      const response = await fetch(`http://127.0.0.1:5001/api/chat/${chatId}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      console.log(`Chat ${chatId} deleted successfully.`);

      // Update local state to remove the deleted chat
      setChats(prevChats => prevChats.filter(chat => chat.id !== chatId));

      // Optional: If the deleted chat was the current one, clear the chat panel in App
      // You could call onChatSelected(null) here, or let App handle it based on its currentChatId state
      if (currentChatId === chatId) {
          // Notify parent that the current chat was deleted (it might want to clear the panel)
          // For now, just clear the local state. App might need logic to handle this if currentChatId is deleted.
          onChatSelected(null as any); // or pass undefined, depending on how App handles it
      }

      setOpenMenuId(null); // Close the menu after deletion

    } catch (err) {
      console.error(`Failed to delete chat ${chatId}:`, err);
      alert("Failed to delete the chat. Please try again.");
    }
  };

  // --- Handle Rename Action (Placeholder) ---
  const handleRename = (chatId: string) => {
    console.log(`Rename action for chat ${chatId} clicked. (Not implemented yet)`);
    // You can implement renaming logic here later
    // e.g., open a prompt, call an API to update the title
    setOpenMenuId(null);
  };

  if (loading) {
    return <div className="h-full flex items-center justify-center bg-gray-800 text-gray-400">Loading chats...</div>;
  }

  if (error) {
    return <div className="h-full flex items-center justify-center bg-gray-800 text-red-500 p-4 text-center">{error}</div>;
  }

  return (
    <div className="h-full flex flex-col bg-gray-800 relative"> {/* Added relative positioning for menu */}
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
      <div className="p-4 flex-grow overflow-y-auto">
        <h2 className="text-xs uppercase text-gray-400 mb-2">HISTORY</h2>
        <ul className="space-y-1">
          {chats.length === 0 ? (
            <li className="text-gray-500 text-sm">No chats yet.</li>
          ) : (
            chats.map((chat) => (
              <li key={chat.id} className="relative"> {/* Added relative for menu positioning */}
                <div
                  onClick={() => handleChatClick(chat.id)}
                  className={`flex justify-between items-center w-full text-left p-2 rounded transition-colors ${
                    currentChatId === chat.id ? 'bg-gray-600' : 'hover:bg-gray-700'
                  }`}
                >
                  <div className="flex items-center">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h8v6H8v-6z" />
                    </svg>
                    <span className="truncate max-w-[calc(100%-20px)]">{chat.title}</span> {/* Truncate title if too long */}
                  </div>
                  {/* Menu Button - Only show on hover */}
                  <button
                    onClick={(e) => handleMenuClick(e, chat.id)}
                    className="p-1 rounded hover:bg-gray-600 opacity-0 group-hover:opacity-100 transition-opacity" // Use group-hover if parent has 'group' class
                  >
                    ...
                  </button>
                </div>
                {/* Hover Menu */}
                {openMenuId === chat.id && (
                  <div
                    ref={menuRef} // Attach ref for click-outside detection
                    className="absolute right-0 mt-1 w-32 bg-gray-700 border border-gray-600 rounded shadow-lg z-10"
                    style={{ top: `${menuPosition.top - 120}px`, left: `${menuPosition.left - 128}px` }} // Position the menu, adjust width (128px) if needed
                  >
                    <ul>
                      {/* <li>
                        <button
                          onClick={() => handleRename(chat.id)}
                          className="block w-full text-left px-4 py-2 hover:bg-gray-600"
                        >
                          Rename
                        </button>
                      </li> */}
                      <li>
                        <button
                          onClick={() => handleDelete(chat.id)}
                          className="block w-full text-left px-4 py-2 text-red-500 hover:bg-red-900/30" // Red hover for delete
                        >
                          Delete
                        </button>
                      </li>
                    </ul>
                  </div>
                )}
              </li>
            ))
          )}
        </ul>
      </div>
    </div>
  );
};

export default Sidebar;
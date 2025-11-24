// src/components/ChatPanel.tsx
import React, { useState } from 'react';

interface Message {
  id: number;
  sender: 'user' | 'bot';
  text: string;
  thinkingProcess?: string[]; // Optional for bot messages
}

const ChatPanel: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      sender: 'bot',
      text: "Hello! I'm Group42 AI. Which retrieval model should I use?",
    },
    {
      id: 2,
      sender: 'user',
      text: "The second place finisher of the 2011 Gran Premio Santander d'Italia drove for who when he won the 2009 FIA Formula One World Championship?",
    },
    {
      id: 3,
      sender: 'bot',
      text: "The second place finisher of the 2011 Gran Premio Santander d'Italia was Jenson Button.",
      thinkingProcess: [
        "Starting multi-hop query processing...",
        "Analyzed intent: Decomposed into 2 sub-questions:",
        "  • Who was the second place finisher of the 2011 Gran Premio Santander d'Italia?",
        "  • Which team did this person drive for when they won the 2009 FIA Formula One World Championship?",
        "Step 1: Searching for \"Who was the second place finisher of the 2011 Gran Premio Santander d'Italia?\"",
        "Found: Jenson Button",
        "Step 2: Searching for \"Who was the second place finisher of the 2011 Gran Premio Santander d'Italia?\"",
        "Found: Jenson Button",
      ],
    },
  ]);

  return (
    <div className="space-y-4">
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
            <p>{msg.text}</p>
          </div>
        </div>
      ))}
    </div>
  );
};

export default ChatPanel;
-- DDL.sql (Updated for UUIDs)

-- Table to store chat sessions
CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY, -- Changed to TEXT to store UUID string
    title TEXT NOT NULL DEFAULT 'New Chat',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Table to store messages within a chat session
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY, -- Changed to TEXT to store UUID string
    session_id TEXT NOT NULL, -- Changed to TEXT to match the foreign key
    sender TEXT NOT NULL CHECK(sender IN ('user', 'bot')),
    content TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    thinking_process TEXT,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
);

-- Optional: Index for faster retrieval of messages by session
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);

-- Optional: Index for faster retrieval of sessions (e.g., for history list)
CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated_at ON chat_sessions(updated_at DESC);
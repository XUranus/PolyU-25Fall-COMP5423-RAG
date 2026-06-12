---
title: API Endpoints
sidebar_position: 2
description: Complete reference for all RAG42 API endpoints with request/response examples.
---

# API Endpoints

## Health Check

Check whether the server is running and the RAG pipeline is initialized.

| | |
|---|---|
| **Method** | `GET` |
| **URL** | `/api/health` |
| **Request Body** | None |

### Response (200)

```json
{
  "ok": true,
  "storage": "/app/storage",
  "cache": "/app/cache",
  "logger": "/app/storage/rag.log",
  "db": "/app/storage/chat_history.db",
  "ready": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `ok` | boolean | Server is running |
| `storage` | string | Path to the storage directory |
| `cache` | string | Path to the cache directory |
| `logger` | string | Path to the log file |
| `db` | string | Path to the SQLite database |
| `ready` | boolean | Whether the RAG pipeline has finished initializing |

---

## List Chat Sessions

Retrieve all chat sessions, ordered by most recently updated.

| | |
|---|---|
| **Method** | `GET` |
| **URL** | `/api/chats/list` |
| **Request Body** | None |

### Response (200)

```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "title": "What is photosynthesis?",
    "updated_at": "2025-11-15 10:30:00"
  },
  {
    "id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
    "title": "New Chat",
    "updated_at": "2025-11-15 09:00:00"
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | string (UUID) | Unique chat session identifier |
| `title` | string | Chat title (auto-set from first message, truncated to 20 chars) |
| `updated_at` | string (datetime) | Last update timestamp |

### Error Response (500)

```json
{
  "error": "Failed to retrieve chat history"
}
```

---

## Create New Chat

Create a new empty chat session.

| | |
|---|---|
| **Method** | `POST` |
| **URL** | `/api/chats/new` |
| **Request Body** | None |

### Response (201)

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "title": "New Chat"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | string (UUID) | Unique identifier for the new chat session |
| `title` | string | Default title is `"New Chat"` |

### Error Response (500)

```json
{
  "error": "Failed to create new chat"
}
```

---

## Delete Chat

Delete a chat session and all its messages (via CASCADE delete).

| | |
|---|---|
| **Method** | `DELETE` |
| **URL** | `/api/chat/<chat_id>` |
| **Request Body** | None |

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `chat_id` | string (UUID) | The chat session to delete |

### Response (200)

```json
{
  "message": "Chat deleted successfully"
}
```

### Error Responses

| Status | Body | Condition |
|--------|------|-----------|
| 404 | `{"error": "Chat session not found"}` | Invalid chat_id |
| 500 | `{"error": "Failed to delete chat"}` | Server error |

---

## Delete Message

Delete a single message by its ID.

| | |
|---|---|
| **Method** | `DELETE` |
| **URL** | `/api/message/<message_id>` |
| **Request Body** | None |

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `message_id` | string (UUID) | The message to delete |

### Response (200)

```json
{
  "message": "message deleted successfully"
}
```

### Error Response (500)

```json
{
  "error": "Failed to delete chat"
}
```

---

## Get Messages

Retrieve all messages in a chat session, ordered by timestamp (oldest first).

| | |
|---|---|
| **Method** | `GET` |
| **URL** | `/api/chat/<chat_id>/messages` |
| **Request Body** | None |

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `chat_id` | string (UUID) | The chat session to retrieve messages from |

### Response (200)

```json
[
  {
    "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "sender": "user",
    "content": "What is photosynthesis?",
    "timestamp": "2025-11-15 10:30:00",
    "thinking_process": null
  },
  {
    "id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
    "sender": "bot",
    "content": "Photosynthesis is the process by which plants convert sunlight into energy...",
    "timestamp": "2025-11-15 10:30:05",
    "thinking_process": [
      {
        "step": "sub_question",
        "query": "What is photosynthesis?",
        "retrieved_docs": [
          {"id": "doc_123", "title": "Photosynthesis", "score": 0.89}
        ]
      }
    ]
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | string (UUID) | Message identifier |
| `sender` | string | `"user"` or `"bot"` |
| `content` | string | Message text |
| `timestamp` | string (datetime) | When the message was created |
| `thinking_process` | array or null | RAG reasoning steps (only for bot messages, null for user messages) |

### Error Responses

| Status | Body | Condition |
|--------|------|-----------|
| 404 | `{"error": "Chat session not found"}` | Invalid chat_id |
| 500 | `{"error": "Failed to retrieve messages for this chat"}` | Server error |

---

## Send Message

Send a user message and receive a RAG-generated response. This is the main endpoint for interacting with the RAG system.

| | |
|---|---|
| **Method** | `POST` |
| **URL** | `/api/chat/<chat_id>/messages` |
| **Content-Type** | `application/json` |

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `chat_id` | string (UUID) | The chat session to send a message to |

### Request Body

```json
{
  "message": "What is the capital of France?",
  "model_name": "Qwen/Qwen2.5-0.5B-Instruct"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | Yes | The user's question (max 10,000 characters) |
| `model_name` | string | No | LLM model to use (defaults to `Qwen/Qwen2.5-0.5B-Instruct`) |

### Response (200)

```json
{
  "user_message_id": "c3d4e5f6-a7b8-9012-cdef-123456789012",
  "id": "d4e5f6a7-b8c9-0123-defa-234567890123",
  "sender": "bot",
  "content": "The capital of France is Paris.",
  "thinking_process": [
    {
      "step": "sub_question",
      "query": "capital of France",
      "retrieved_docs": [
        {"id": "doc_456", "title": "France", "score": 0.95}
      ],
      "answer": "Paris"
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `user_message_id` | string (UUID) | ID of the stored user message |
| `id` | string (UUID) | ID of the bot's response message |
| `sender` | string | Always `"bot"` |
| `content` | string | The generated answer |
| `thinking_process` | array | Step-by-step RAG reasoning with retrieved documents |

### Error Responses

| Status | Body | Condition |
|--------|------|-----------|
| 400 | `{"error": "Message content is required"}` | Empty message |
| 400 | `{"error": "Message too long (max 10000 characters)"}` | Message exceeds limit |
| 404 | `{"error": "Chat session not found"}` | Invalid chat_id |
| 500 | `{"error": "RAG module not initialized"}` | RAG pipeline not ready |
| 500 | `{"error": "Failed to process message"}` | Server error |

:::tip Multi-turn Conversations
The server automatically retrieves previous messages in the chat session and passes them as conversation history to the RAG pipeline. This enables multi-turn dialogue where follow-up questions can reference earlier context.
:::

:::info Automatic Title Generation
When the second message is stored (the first user message + first bot response), the chat title is automatically set to the first 20 characters of the user's message.
:::

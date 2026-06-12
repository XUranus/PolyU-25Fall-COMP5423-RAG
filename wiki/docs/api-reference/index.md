---
title: API Overview
sidebar_position: 1
description: Overview of the RAG42 Flask backend API, request/response formats, and error handling.
---

# API Overview

The RAG42 backend is a Flask server that provides a RESTful API for managing chat sessions, sending messages, and receiving RAG-generated responses. It uses SQLite for persistent storage.

## Base URL

```
http://localhost:5000
```

The host and port are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG42_BACKEND_HOST` | `0.0.0.0` | Bind address |
| `RAG42_BACKEND_PORT` | `5000` | Listen port |

## Request and Response Format

All endpoints use **JSON** for both requests and responses.

### Request Headers

When sending a request body, include:

```
Content-Type: application/json
```

### Response Structure

Successful responses return JSON with appropriate HTTP status codes:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "title": "New Chat"
}
```

Status code `200` for successful reads, `201` for successful creates.

## Error Handling

Errors follow a consistent pattern:

```json
{
  "error": "Human-readable error message"
}
```

| Status Code | Meaning | Example |
|-------------|---------|---------|
| `200` | Success | Request completed |
| `201` | Created | New resource created |
| `400` | Bad Request | Missing or invalid input |
| `404` | Not Found | Chat session or message does not exist |
| `500` | Internal Server Error | Server-side failure |

### Example Error Response

```json
{
  "error": "Chat session not found"
}
```

:::tip
The server logs detailed error information (including stack traces) to the log file at `{RAG42_STORAGE_DIR}/rag.log`. Check this file when debugging API errors.
:::

## CORS

Cross-Origin Resource Sharing (CORS) is enabled. By default, all origins are allowed (`*`). To restrict origins, set the `RAG42_CORS_ORIGINS` environment variable as a comma-separated list:

```
RAG42_CORS_ORIGINS=http://localhost:3000,https://myapp.example.com
```

## RAG Initialization

The RAG pipeline initializes asynchronously when the server starts. The `/api/health` endpoint reports whether initialization is complete via the `ready` field:

```json
{
  "ok": true,
  "ready": false,
  "storage": "/app/storage",
  "cache": "/app/cache"
}
```

:::warning
The `/api/chat/<id>/messages` POST endpoint returns a `500` error if RAG is not yet initialized. Wait for `ready: true` before sending messages.
:::

## Endpoints at a Glance

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | [`/api/health`](./endpoints.md#health-check) | Server health and RAG status |
| GET | [`/api/chats/list`](./endpoints.md#list-chat-sessions) | List all chat sessions |
| POST | [`/api/chats/new`](./endpoints.md#create-new-chat) | Create a new chat session |
| DELETE | [`/api/chat/<chat_id>`](./endpoints.md#delete-chat) | Delete a chat session |
| DELETE | [`/api/message/<message_id>`](./endpoints.md#delete-message) | Delete a single message |
| GET | [`/api/chat/<chat_id>/messages`](./endpoints.md#get-messages) | Get all messages in a chat |
| POST | [`/api/chat/<chat_id>/messages`](./endpoints.md#send-message) | Send a message and get a RAG response |

See the [Endpoints](./endpoints.md) page for full details on each endpoint.

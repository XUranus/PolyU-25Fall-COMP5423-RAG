#!/usr/bin/env python3

from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import json
import time
import uuid
from datetime import datetime
from logging import DEBUG, INFO, WARNING, ERROR, basicConfig, getLogger

# selfdefined modules
from rag import RAGSystem

logger = getLogger(__name__)
logger.setLevel(INFO)


logger.info("Initializing RAG system...")
rag = RAGSystem()
rag.bootstrap()


app = Flask(__name__)
CORS(app)

DATABASE = 'chat_history.db'

# --- Helper Functions ---

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row # This allows us to fetch rows as dictionaries
    return conn

def init_db():
    """Initializes the database with the required tables."""
    with app.app_context():
        db = get_db_connection()
        with app.open_resource('DDL.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()
        db.close()


# --- API Endpoints ---


@app.route('/api/chats/list', methods=['GET'])
def get_chat_history():
    """
    Retrieves a list of all chat sessions for the sidebar.
    Returns a list of dictionaries containing id and title.
    """
    try:
        conn = get_db_connection()
        chats = conn.execute('''
            SELECT id, title, updated_at
            FROM chat_sessions
            ORDER BY updated_at DESC
        ''').fetchall()
        conn.close()

        # Convert Row objects to dictionaries
        chat_list = [dict(chat) for chat in chats]
        return jsonify(chat_list), 200
    except Exception as e:
        print(f"Error fetching chat history: {e}")
        return jsonify({'error': 'Failed to retrieve chat history'}), 500


@app.route('/api/chats/new', methods=['POST'])
def create_new_chat():
    """
    Creates a new chat session using a UUID.
    """
    try:
        conn = get_db_connection()
        # Generate a new UUID for the session
        new_chat_id = str(uuid.uuid4())
        # Insert a new session with the generated UUID
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO chat_sessions (id, title)
            VALUES (?, ?)
        ''', (new_chat_id, 'New Chat')) # Explicitly set the ID and default title
        conn.commit()
        conn.close()

        # Return the ID of the new chat
        return jsonify({'id': new_chat_id, 'title': 'New Chat'}), 201
    except Exception as e:
        print(f"Error creating new chat: {e}")
        return jsonify({'error': 'Failed to create new chat'}), 500


@app.route('/api/chat/<string:chat_id>', methods=['DELETE'])
def delete_chat(chat_id : str):
    """
    Deletes a specific chat session and its messages.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Check if the chat session exists
        session_check = cur.execute('SELECT id FROM chat_sessions WHERE id = ?', (chat_id,)).fetchone()
        if not session_check:
            conn.close()
            return jsonify({'error': 'Chat session not found'}), 404

        # Delete the session (messages will be deleted automatically due to CASCADE)
        cur.execute('DELETE FROM chat_sessions WHERE id = ?', (chat_id,))
        conn.commit()
        conn.close()

        # If the deleted chat was the current one, the frontend might want to clear the chat panel
        # Returning a success message is sufficient here
        return jsonify({'message': 'Chat deleted successfully'}), 200

    except Exception as e:
        print(f"Error deleting chat {chat_id}: {e}")
        conn.rollback() # Rollback in case of error
        return jsonify({'error': 'Failed to delete chat'}), 500


@app.route('/api/chat/<string:chat_id>/messages', methods=['GET'])
def get_messages(chat_id : str):
    """
    Retrieves all messages for a specific chat session identified by UUID.
    """
    try:
        conn = get_db_connection()
        messages = conn.execute('''
            SELECT id, sender, content, timestamp, thinking_process, retrieved_docs
            FROM messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
        ''', (chat_id,)).fetchall() # Use the UUID string as the parameter
        conn.close()

        message_list = [dict(msg) for msg in messages]
        return jsonify(message_list), 200
    except Exception as e:
        print(f"Error fetching messages for chat {chat_id}: {e}")
        return jsonify({'error': 'Failed to retrieve messages for this chat'}), 500


@app.route('/api/chat/<string:chat_id>/messages', methods=['POST'])
def send_message(chat_id : str):
    """
    Handles a new user message for a specific chat UUID, calls the RAG system,
    stores both messages, and returns the bot's response.
    Expects JSON: { "message": "user's query" }
    """
    try:
        user_data = request.get_json()
        user_message = user_data.get('message', '').strip()

        if not user_message:
            return jsonify({'error': 'Message content is required'}), 400

        # Validate that the chat_id exists before proceeding
        conn = get_db_connection()
        session_check = conn.execute('SELECT id FROM chat_sessions WHERE id = ?', (chat_id,)).fetchone()
        if not session_check:
            conn.close()
            return jsonify({'error': 'Chat session not found'}), 404

        # 1. Store the user's message with a UUID
        user_message_id = str(uuid.uuid4())
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO messages (id, session_id, sender, content)
            VALUES (?, ?, 'user', ?)
        ''', (user_message_id, chat_id, user_message)) # Use generated UUID and provided chat_id

        # 2. Call your RAG logic here (pseudo-code placeholder)
        rag_response = {
            "answer": f"Echoing: {user_message}", # Placeholder
            "thinking_process": ["Simulated thinking step 1.", "Simulated thinking step 2."], # Placeholder
            "retrieved_docs": [["doc_123", 0.85], ["doc_456", 0.79]] # Placeholder
        }

        # 3. Store the bot's response message with a UUID
        bot_message_id = str(uuid.uuid4())
        cur.execute('''
            INSERT INTO messages (id, session_id, sender, content, thinking_process, retrieved_docs)
            VALUES (?, ?, 'bot', ?, ?, ?)
        ''', (bot_message_id, chat_id, rag_response['answer'], json.dumps(rag_response['thinking_process']), json.dumps(rag_response['retrieved_docs'])))

        # Optional: Update the chat title based on the first user message or a summary
        existing_messages = conn.execute('''
            SELECT id FROM messages WHERE session_id = ? LIMIT 2
        ''', (chat_id,)).fetchall()
        if len(existing_messages) == 2: # User message was just added, making it 2 total
             title = (user_message[:20] + '..') if len(user_message) > 20 else user_message
             cur.execute('UPDATE chat_sessions SET title = ? WHERE id = ?', (title, chat_id))

        conn.commit()
        conn.close()

        # 4. Return the bot's response to the frontend
        return jsonify({
            'id': bot_message_id, # Return the ID of the bot's message just inserted
            'sender': 'bot',
            'content': rag_response['answer'],
            'thinking_process': rag_response['thinking_process'],
            'retrieved_docs': rag_response['retrieved_docs']
        }), 200

    except Exception as e:
        print(f"Error processing message for chat {chat_id}: {e}")
        # Consider rolling back the transaction if both messages should be atomic
        if 'conn' in locals():
            conn.rollback()
        return jsonify({'error': 'Failed to process message'}), 500


# --- Main Execution Block ---
if __name__ == '__main__':
    # Initialize the database when the script is run directly
    init_db()
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5001)
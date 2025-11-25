#!/usr/bin/env python3

from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import json
import time
import uuid
from datetime import datetime
import logging
import os

def setup_logger(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding handlers multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    # console_handler = logging.StreamHandler(sys.stdout)
    # console_handler.setLevel(level)
    # console_format = logging.Formatter('%(levelname)s - %(message)s')
    # console_handler.setFormatter(console_format)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] [%(message)s]')
    file_handler.setFormatter(file_format)

    # logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# Configuration
STORAGE_DIR = os.getenv('RAG42_STORAGE_DIR', './storage')
os.makedirs(STORAGE_DIR, exist_ok = True)
DATABASE_PATH = os.path.join(STORAGE_DIR, 'chat_history.db')

# Logger
APPNAME = "RAG42"
logger = setup_logger(APPNAME, 'server.log', level=logging.INFO)
logger.info("== RAG42 Server Starting ==")
logger.info(f"Using storage directory: {STORAGE_DIR}")

# RAG modules
logger.info("import RAG modules...")
now = time.time()
from rag_pipeline import RAGPipeline
from hybrid_retriever import HybridRetriever
from qwen_generator import QwenGenerator
logger.info(f"RAG modules loaded. ({(time.time() - now):.2f} seconds)")

logger.info("Initialize RAG modules...")
now = time.time()
retriever = HybridRetriever(collection_path="izhx/COMP5423-25Fall-HQ-small")
generator = QwenGenerator(model_name="Qwen/Qwen2.5-0.5B-Instruct")
rag_pipeline = RAGPipeline(retriever=retriever, generator=generator)
logger.info(f"RAG modules initialized. ({(time.time() - now):.2f} seconds)")


logger.info("Starting Flask app...")
app = Flask(__name__)
CORS(app)



# --- Helper Functions ---

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row # This allows us to fetch rows as dictionaries
    return conn


def init_db():
    """Initializes the database with the required tables."""
    with app.app_context():
        db = get_db_connection()
        with app.open_resource('db_init.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()
        db.close()


# --- API Endpoints ---

@app.route('/api/health')
def api_health():
    return jsonify({
        "status": "ok",
        "service": "flask-backend",
        "storage_dir": RAG42_STORAGE_DIR
    })


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
        logger.error(f"Error fetching chat history: {e}")
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
        logger.error(f"Error creating new chat: {e}")
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
        logger.error(f"Error deleting chat {chat_id}: {e}")
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
            SELECT id, sender, content, timestamp, thinking_process
            FROM messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
        ''', (chat_id,)).fetchall() # Use the UUID string as the parameter
        conn.close()

        message_list = [dict(msg) for msg in messages]
        return jsonify(message_list), 200
    except Exception as e:
        logger.error(f"Error fetching messages for chat {chat_id}: {e}")
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

        # 2. RAG logic here
        # --- Run RAG Pipeline ---
        # Note: For now, session_history is passed as None.
        # You can retrieve it from the DB later if implementing multi-turn.
        now = time.time()
        logger.info(f"processing message for [{user_message_id}]:[{user_message[:50]}...]")
        rag_result = rag_pipeline.run(query=user_message, session_history=None)
        bot_response = rag_result["answer"]
        thinking_process = rag_result["thinking_process"] # [str]
        logger.info(f"Query {user_message_id} took {(time.time() - now):.2f} seconds.")

        rag_response = {
            "answer": bot_response,
            "thinking_process": thinking_process
        }

        # 3. Store the bot's response message with a UUID
        bot_message_id = str(uuid.uuid4())
        cur.execute('''
            INSERT INTO messages (id, session_id, sender, content, thinking_process)
            VALUES (?, ?, 'bot', ?, ?)
        ''', (bot_message_id, chat_id, rag_response['answer'], json.dumps(rag_response['thinking_process'])))

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
            'thinking_process': rag_response['thinking_process']
        }), 200

    except Exception as e:
        logger.error(f"Error processing message for chat {chat_id}: {e}")
        # Consider rolling back the transaction if both messages should be atomic
        if 'conn' in locals():
            conn.rollback()
        return jsonify({'error': 'Failed to process message'}), 500


# --- Main Execution Block ---
if __name__ == '__main__':
    # Initialize the database when the script is run directly
    init_db()
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)
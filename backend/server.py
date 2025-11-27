#!/usr/bin/env python3
# server.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import json
import time
import uuid
from datetime import datetime
import logging
import os
import traceback

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
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
RAG42_STORAGE_DIR = os.getenv('RAG42_STORAGE_DIR', './storage')
RAG42_CACHE_DIR=os.getenv('RAG42_CACHE_DIR', './cache')
RAG42_BACKEND_PORT = os.getenv('RAG42_BACKEND_PORT', '5000')
RAG42_BACKEND_HOST = os.getenv('RAG42_BACKEND_HOST', '0.0.0.0')
os.makedirs(RAG42_STORAGE_DIR, exist_ok = True)
os.makedirs(RAG42_CACHE_DIR, exist_ok = True)
DATABASE_PATH = os.path.join(RAG42_STORAGE_DIR, 'chat_history.db')
LOGGER_PATH = os.path.join(RAG42_STORAGE_DIR, 'rag.log')

# Logger
APPNAME = "RAG42"
logger = setup_logger(APPNAME, LOGGER_PATH, level=logging.INFO)
logger.info("== RAG42 Server Starting ==")
logger.info(f"Using storage directory: {RAG42_STORAGE_DIR}")
logger.info(f"Using cache directory: {RAG42_CACHE_DIR}")
logger.info(f"Using address: {RAG42_BACKEND_HOST}:{RAG42_BACKEND_PORT}")
logger.info(f"Using database: {DATABASE_PATH}")
logger.info(f"Using logger: {LOGGER_PATH}")

# RAG modules
logger.info("import RAG modules...")
now = time.time()
from rag_pipeline import RAGPipeline
from hybrid_retriever import HybridRetriever
logger.info(f"RAG modules loaded. ({(time.time() - now):.2f} seconds)")

logger.info("Initialize RAG modules...")
now = time.time()
retriever = HybridRetriever(collection_path="izhx/COMP5423-25Fall-HQ-small")
rag_pipeline = RAGPipeline(retriever=retriever)
rag_pipeline.init_generator(model_name=DEFAULT_MODEL)
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
        logger.error(traceback.format_exc())
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
        logger.error(traceback.format_exc())
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
        logger.error(traceback.format_exc())
        conn.rollback() # Rollback in case of error
        return jsonify({'error': 'Failed to delete chat'}), 500


@app.route('/api/message/<string:message_id>', methods=['DELETE'])
def delete_message(message_id : str):
    """
    Deletes a specific message.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Delete the session (messages will be deleted automatically due to CASCADE)
        cur.execute('DELETE FROM messages WHERE id = ?', (message_id,))
        conn.commit()
        conn.close()
        return jsonify({'message': 'message deleted successfully'}), 200

    except Exception as e:
        logger.error(f"Error deleting message {message_id}: {e}")
        logger.error(traceback.format_exc())
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
        logger.error(traceback.format_exc())
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
        model_name = user_data.get('model_name', DEFAULT_MODEL).strip()
        logger.debug(f"user send new query using model: {model_name}")

        if not user_message:
            return jsonify({'error': 'Message content is required'}), 400

        # Validate that the chat_id exists before proceeding
        conn = get_db_connection()
        session_check = conn.execute('SELECT id FROM chat_sessions WHERE id = ?', (chat_id,)).fetchone()
        if not session_check:
            conn.close()
            return jsonify({'error': 'Chat session not found'}), 404
        
        # 1. Collect history dislogues for Multi-turn
        history_dialogues = conn.execute('''
            SELECT sender, content 
            FROM messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
        ''', (chat_id,)).fetchall()
        history_dialogues = [dict(msg) for msg in history_dialogues]
        history_dialogues
        logger.debug(f'Collected history dialogues: {history_dialogues}')

        # 2. Store the user's message with a UUID
        user_message_id = str(uuid.uuid4())
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO messages (id, session_id, sender, content)
            VALUES (?, ?, 'user', ?)
        ''', (user_message_id, chat_id, user_message)) # Use generated UUID and provided chat_id

        # 3. RAG logic here
        # --- Run RAG Pipeline ---
        # Note: For now, session_history is passed as None.
        # You can retrieve it from the DB later if implementing multi-turn.
        now = time.time()
        logger.info(f"processing message for [{user_message_id}]:[{user_message[:50]}...]")
        rag_result = rag_pipeline.run(
            query = user_message,
            session_history = history_dialogues,
            model_name=model_name)
        bot_response = rag_result["answer"]
        thinking_process = rag_result["thinking_process"] # [str]
        logger.info(f"Query {user_message_id} took {(time.time() - now):.2f} seconds.")
        logger.debug(f"Bot response: {bot_response}")
        logger.debug(f"Thinking process: {thinking_process}")

        # 4. Store the bot's response message with a UUID
        bot_message_id = str(uuid.uuid4())
        cur.execute('''
            INSERT INTO messages (id, session_id, sender, content, thinking_process)
            VALUES (?, ?, 'bot', ?, ?)
        ''', (bot_message_id, chat_id, bot_response, json.dumps(thinking_process)))

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
            'user_message_id' : user_message_id,
            'id': bot_message_id, # Return the ID of the bot's message just inserted
            'sender': 'bot',
            'content': bot_response,
            'thinking_process': thinking_process
        }), 200

    except Exception as e:
        logger.error(f"Error processing message for chat {chat_id}: {e}")
        logger.error(traceback.format_exc())
        # Consider rolling back the transaction if both messages should be atomic
        if 'conn' in locals():
            conn.rollback()
        return jsonify({'error': 'Failed to process message'}), 500


# --- Main Execution Block ---
if __name__ == '__main__':
    # Initialize the database when the script is run directly
    init_db()
    # Run the Flask app
    app.run(debug=True, host=RAG42_BACKEND_HOST, port=int(RAG42_BACKEND_PORT), use_reloader=False)
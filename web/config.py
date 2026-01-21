"""
Web configuration for Violence Detection System.
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent
WEB_DIR = Path(__file__).parent

# Upload settings
UPLOAD_FOLDER = WEB_DIR / 'static' / 'uploads'
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Model settings
MODEL_PATH = BASE_DIR / 'models' / 'checkpoints' / 'best_model.h5'
CONFIG_PATH = BASE_DIR / 'config.yaml'

# Database settings (SQLite for now, easy to migrate to SQL Server)
DATABASE_PATH = WEB_DIR / 'detections.db'

# Flask settings
SECRET_KEY = 'your-secret-key-change-in-production'
DEBUG = True

# CORS settings
CORS_ORIGINS = '*'  # Change to specific domain in production

# WebSocket settings
SOCKETIO_ASYNC_MODE = 'eventlet'

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

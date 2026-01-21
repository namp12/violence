"""
Flask Web Application for Violence Detection System.
Main application entry point.
"""
import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import tensorflow as tf

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from web.config import *
from web.database import Database
from web.api.video_handler import VideoHandler
from web.api.webcam_handler import WebcamHandler
from web.email_notifier import EmailNotifier
from src.utils.config import get_config

# Initialize Flask app
base_dir = Path(__file__).parent.absolute()
app = Flask(__name__, 
            static_folder=str(base_dir / 'static'),
            template_folder=str(base_dir / 'templates'))
app.config['SECRET_KEY'] = SECRET_KEY
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Enable CORS
CORS(app, origins=CORS_ORIGINS)

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins=CORS_ORIGINS, async_mode=SOCKETIO_ASYNC_MODE)

# Initialize database
db = Database(DATABASE_PATH)

# Initialize email notifier
email_notifier = EmailNotifier()

# Load model and config (lazy loading)
model = None
config = None
video_handler = None
webcam_handler = None

def load_ml_components():
    """Load ML model and handlers (lazy loading)."""
    global model, config, video_handler, webcam_handler
    
    if model is None:
        print("Loading ML components...")
        
        # Load config
        config = get_config(str(CONFIG_PATH))
        
        # Load model
        if MODEL_PATH.exists():
            model = tf.keras.models.load_model(str(MODEL_PATH))
            print(f"✓ Model loaded from {MODEL_PATH}")
        else:
            print(f"⚠ Model not found at {MODEL_PATH}")
            print("  Please complete training first!")
            return False
        
        # Initialize handlers
        video_handler = VideoHandler(model, config, UPLOAD_FOLDER)
        webcam_handler = WebcamHandler(model, config)
        
        print("✓ ML components loaded successfully")
    
    return True


# ==================== ROUTES ====================

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    """API health check - responds immediately without loading model."""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'database_connected': DATABASE_PATH.exists()
    })

@app.route('/api/upload', methods=['POST'])
def api_upload():
    """Upload video file."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv'}), 400
    
    # Load ML components
    if not load_ml_components():
        return jsonify({'error': 'ML model not ready. Please complete training first.'}), 503
    
    # Save file
    filepath = video_handler.save_upload(file)
    if filepath is None:
        return jsonify({'error': 'Failed to save file'}), 500
    
    filename = filepath.name
    
    return jsonify({
        'success': True,
        'filename': filename,
        'message': 'Video uploaded successfully'
    })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Predict violence in uploaded video."""
    data = request.get_json()
    
    if not data or 'filename' not in data:
        return jsonify({'error': 'No filename provided'}), 400
    
    filename = secure_filename(data['filename'])
    filepath = UPLOAD_FOLDER / filename
    
    if not filepath.exists():
        return jsonify({'error': 'Video file not found'}), 404
    
    # Load ML components
    if not load_ml_components():
        return jsonify({'error': 'ML model not ready'}), 503
    
    # Predict
    result = video_handler.predict_video(filepath)
    
    if result is None:
        return jsonify({'error': 'Failed to process video'}), 500
    
    # Save to database
    db.add_detection(
        video_name=filename,
        prediction=result['prediction'],
        confidence=result['confidence'],
        source='upload',
        video_path=str(filepath)
    )
    
    # Send email alert if violent
    if result['is_violent']:
        email_notifier.send_alert(
            video_name=filename,
            confidence=result['confidence'],
            source='upload'
        )
    
    return jsonify({
        'success': True,
        'result': result
    })

@app.route('/api/history')
def api_history():
    """Get detection history."""
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    source = request.args.get('source', None)
    
    detections = db.get_detections(limit=limit, offset=offset, source=source)
    
    return jsonify({
        'success': True,
        'detections': detections,
        'count': len(detections)
    })

@app.route('/api/statistics')
def api_statistics():
    """Get detection statistics."""
    days = request.args.get('days', 7, type=int)
    
    stats = db.get_statistics(days=days)
    totals = db.get_total_counts()
    
    return jsonify({
        'success': True,
        'daily_stats': stats,
        'totals': totals
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ==================== SOCKET.IO EVENTS ====================

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print(f"Client connected: {request.sid}")
    emit('connected', {'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print(f"Client disconnected: {request.sid}")
    if webcam_handler:
        webcam_handler.remove_buffer(request.sid)

@socketio.on('start_webcam')
def handle_start_webcam():
    """Handle webcam start request."""
    if not load_ml_components():
        emit('error', {'message': 'ML model not ready'})
        return
    
    print(f"Webcam started for client: {request.sid}")
    emit('webcam_started', {'message': 'Webcam detection started'})

@socketio.on('webcam_frame')
def handle_webcam_frame(data):
    """
    Handle incoming webcam frame.
    
    data should contain:
        - frame: base64 encoded image
    """
    if not load_ml_components():
        emit('error', {'message': 'ML model not ready'})
        return
    
    frame_data = data.get('frame')
    if not frame_data:
        return
    
    # Process frame
    result = webcam_handler.process_frame(frame_data, request.sid)
    
    # Send result if available
    if result:
        # Check if we should trigger an alert (database/email)
        # This prevents spamming for every frame with violence
        should_alert = webcam_handler.should_trigger_alert(
            request.sid, 
            result['is_violent'], 
            result['confidence']
        )
        
        if should_alert:
            # Save to database
            db.add_detection(
                video_name='Webcam',
                prediction=result['prediction'],
                confidence=result['confidence'],
                source='webcam'
            )
            
            # Send email alert
            email_notifier.send_alert(
                video_name='Webcam Live Stream',
                confidence=result['confidence'],
                source='webcam'
            )
        
        # Always emit prediction to update UI in real-time
        emit('prediction', result)

@socketio.on('stop_webcam')
def handle_stop_webcam():
    """Handle webcam stop request."""
    if webcam_handler:
        webcam_handler.clear_buffer(request.sid)
    
    print(f"Webcam stopped for client: {request.sid}")
    emit('webcam_stopped', {'message': 'Webcam detection stopped'})


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


# ==================== MAIN ====================

if __name__ == '__main__':
    # Get port from environment variable (Railway, Render) or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    print("=" * 70)
    print("Violence Detection Web Application")
    print("=" * 70)
    print(f"\nServer starting on http://0.0.0.0:{port}")
    print("\nMake sure:")
    print("  1. Training is complete (model file exists)")
    print("  2. Web dependencies installed: pip install -r requirements_web.txt")
    print("\nPress Ctrl+C to stop the server\n")
    
    # Run with SocketIO
    socketio.run(app, debug=DEBUG, host='0.0.0.0', port=port)

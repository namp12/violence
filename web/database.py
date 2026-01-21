"""
Database models and operations for Violence Detection Web App.
Using SQLite for development (easy migration to SQL Server later).
"""
import sqlite3
from datetime import datetime
from pathlib import Path
import json

class Database:
    """Database handler for detections."""
    
    def __init__(self, db_path):
        """Initialize database connection."""
        self.db_path = db_path
        self.init_db()
    
    def get_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return dict-like rows
        return conn
    
    def init_db(self):
        """Initialize database tables."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Create detections table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_name TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                source TEXT NOT NULL,
                video_path TEXT
            )
        ''')
        
        # Create statistics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                total_detections INTEGER DEFAULT 0,
                violent_count INTEGER DEFAULT 0,
                non_violent_count INTEGER DEFAULT 0,
                UNIQUE(date)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_detection(self, video_name, prediction, confidence, source='upload', video_path=None):
        """
        Add a new detection record.
        
        Args:
            video_name: Name of the video
            prediction: 'Violent' or 'Non-Violent'
            confidence: Confidence score (0-1)
            source: 'upload' or 'webcam'
            video_path: Path to saved video (optional)
            
        Returns:
            ID of inserted record
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO detections (video_name, prediction, confidence, source, video_path)
            VALUES (?, ?, ?, ?, ?)
        ''', (video_name, prediction, confidence, source, video_path))
        
        detection_id = cursor.lastrowid
        
        # Update statistics
        today = datetime.now().date()
        cursor.execute('''
            INSERT INTO statistics (date, total_detections, violent_count, non_violent_count)
            VALUES (?, 1, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                total_detections = total_detections + 1,
                violent_count = violent_count + ?,
                non_violent_count = non_violent_count + ?
        ''', (
            today,
            1 if prediction == 'Violent' else 0,
            0 if prediction == 'Violent' else 1,
            1 if prediction == 'Violent' else 0,
            0 if prediction == 'Violent' else 1
        ))
        
        conn.commit()
        conn.close()
        
        return detection_id
    
    def get_detections(self, limit=50, offset=0, source=None):
        """
        Get detection records.
        
        Args:
            limit: Maximum number of records
            offset: Offset for pagination
            source: Filter by source ('upload' or 'webcam')
            
        Returns:
            List of detection records
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        query = 'SELECT * FROM detections'
        params = []
        
        if source:
            query += ' WHERE source = ?'
            params.append(source)
        
        query += ' ORDER BY timestamp DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        conn.close()
        
        # Convert to list of dicts
        return [dict(row) for row in rows]
    
    def get_statistics(self, days=7):
        """
        Get statistics for the last N days.
        
        Args:
            days: Number of days to fetch
            
        Returns:
            List of daily statistics
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM statistics
            ORDER BY date DESC
            LIMIT ?
        ''', (days,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_total_counts(self):
        """
        Get total counts of detections.
        
        Returns:
            Dict with total, violent, and non-violent counts
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN prediction = 'Violent' THEN 1 ELSE 0 END) as violent,
                SUM(CASE WHEN prediction = 'Non-Violent' THEN 1 ELSE 0 END) as non_violent
            FROM detections
        ''')
        
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else {'total': 0, 'violent': 0, 'non_violent': 0}
    
    def delete_old_records(self, days=30):
        """Delete records older than specified days."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM detections
            WHERE timestamp < datetime('now', '-' || ? || ' days')
        ''', (days,))
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted

from app import db
from datetime import datetime
from sqlalchemy import Text, Float, Integer, String, DateTime

class Analysis(db.Model):
    id = db.Column(Integer, primary_key=True)
    filename = db.Column(String(255), nullable=False)
    file_type = db.Column(String(10), nullable=False)  # 'video', 'audio', or 'image'
    file_path = db.Column(String(500), nullable=False)
    
    # Analysis results
    confidence_score = db.Column(Float, nullable=False)  # 0-100 fake probability
    video_score = db.Column(Float, nullable=True)  # Video analysis score
    audio_score = db.Column(Float, nullable=True)  # Audio analysis score
    
    # Detailed results
    face_consistency_score = db.Column(Float, nullable=True)
    frame_artifacts_score = db.Column(Float, nullable=True)
    spectral_anomaly_score = db.Column(Float, nullable=True)
    frequency_pattern_score = db.Column(Float, nullable=True)
    
    # Image analysis results
    image_score = db.Column(Float, nullable=True)  # Image analysis score
    face_analysis_score = db.Column(Float, nullable=True)
    compression_artifacts_score = db.Column(Float, nullable=True)
    pixel_inconsistency_score = db.Column(Float, nullable=True)
    metadata_score = db.Column(Float, nullable=True)
    
    # FaceForensics++ results
    faceforensics_score = db.Column(Float, nullable=True)  # FaceForensics++ XceptionNet score
    
    # Xception model results
    xception_score = db.Column(Float, nullable=True)  # Xception model score
    
    # Metadata
    file_size = db.Column(Integer, nullable=False)
    duration = db.Column(Float, nullable=True)  # Duration in seconds
    processing_time = db.Column(Float, nullable=False)  # Processing time in seconds
    created_at = db.Column(DateTime, default=datetime.utcnow)
    
    # Analysis details
    analysis_details = db.Column(Text, nullable=True)  # JSON string with detailed results
    
    def __repr__(self):
        return f'<Analysis {self.filename}: {self.confidence_score}% fake>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'file_type': self.file_type,
            'confidence_score': self.confidence_score,
            'video_score': self.video_score,
            'audio_score': self.audio_score,
            'image_score': self.image_score,
            'face_consistency_score': self.face_consistency_score,
            'frame_artifacts_score': self.frame_artifacts_score,
            'spectral_anomaly_score': self.spectral_anomaly_score,
            'frequency_pattern_score': self.frequency_pattern_score,
            'face_analysis_score': self.face_analysis_score,
            'compression_artifacts_score': self.compression_artifacts_score,
            'pixel_inconsistency_score': self.pixel_inconsistency_score,
            'metadata_score': self.metadata_score,
            'faceforensics_score': self.faceforensics_score,
            'xception_score': self.xception_score,
            'file_size': self.file_size,
            'duration': self.duration,
            'processing_time': self.processing_time,
            'created_at': self.created_at.isoformat(),
            'analysis_details': self.analysis_details
        }

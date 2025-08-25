# Deepfake Detector

## Overview

A Flask-based web application for detecting deepfake content in video and audio files. The system uses computer vision techniques for video analysis and spectral analysis for audio detection, combining results through an ensemble scoring approach to provide confidence ratings on whether uploaded media is likely to be authentic or artificially generated.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Backend Framework
- **Flask Web Framework**: Lightweight Python web framework serving as the application backbone
- **SQLAlchemy ORM**: Database abstraction layer with declarative models for data persistence
- **Database**: SQLite for development with PostgreSQL compatibility built-in through environment configuration

### File Processing Pipeline
- **Multi-format Support**: Handles video files (MP4, AVI, MOV, MKV) and audio files (WAV, MP3, M4A, FLAC)
- **File Upload System**: Secure file handling with size limits (100MB) and type validation
- **Audio Extraction**: FFmpeg integration for extracting audio tracks from video files

### Detection Engine Architecture
- **Modular Analysis System**: Separate analyzers for video and audio content
  - `VideoAnalyzer`: Computer vision-based analysis using OpenCV for face detection and frame artifact analysis
  - `AudioAnalyzer`: Spectral analysis using librosa for frequency pattern and anomaly detection
- **Ensemble Scoring**: Weighted combination of multiple analysis results for final confidence scoring
- **Scoring Range**: 0-100 scale where higher scores indicate higher probability of deepfake content

### Data Model
- **Analysis Entity**: Comprehensive storage of detection results including:
  - File metadata (name, type, size, duration)
  - Individual analysis scores (video, audio, face consistency, artifacts)
  - Processing metrics and timestamps
  - Detailed analysis results in JSON format

### Frontend Architecture
- **Bootstrap-based UI**: Dark theme responsive design with modern components
- **Progressive Enhancement**: JavaScript-enhanced file upload with drag-and-drop functionality
- **Chart Visualization**: Chart.js integration for displaying confidence scores and analysis results
- **AJAX Upload**: Asynchronous file processing with progress feedback

### Security Features
- **File Type Validation**: Server-side verification of allowed file extensions
- **Secure Filename Handling**: Werkzeug's secure_filename utility for path traversal protection
- **Upload Size Limits**: Configurable maximum file size restrictions
- **Proxy Protection**: ProxyFix middleware for secure header handling

## External Dependencies

### Core Framework Dependencies
- **Flask**: Web application framework
- **SQLAlchemy**: Database ORM and migrations
- **Werkzeug**: WSGI utilities and security helpers

### Media Processing Libraries
- **OpenCV (cv2)**: Computer vision operations for video analysis
- **librosa**: Audio analysis and feature extraction
- **NumPy**: Numerical computing for data processing

### Optional System Dependencies
- **FFmpeg**: Audio extraction from video files (graceful fallback if unavailable)

### Frontend Libraries (CDN)
- **Bootstrap**: UI framework with dark theme
- **Chart.js**: Data visualization for results display
- **Feather Icons**: Icon library for UI elements

### Development Tools
- **Python Logging**: Built-in debugging and monitoring
- **Environment Configuration**: Database and secret key management through environment variables
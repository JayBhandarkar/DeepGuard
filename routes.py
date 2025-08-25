import os
import time
import json
from flask import render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from app import app, db
from models import Analysis
from detection.enhanced_video_analyzer import EnhancedVideoAnalyzer
from detection.audio_analyzer import AudioAnalyzer
from detection.enhanced_image_analyzer import EnhancedImageAnalyzer
from detection.ensemble_scorer import EnsembleScorer
# from utils.file_handler import FileHandler  # Not currently used

ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'}

def allowed_file(filename, file_type):
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if file_type == 'video':
        return ext in ALLOWED_VIDEO_EXTENSIONS
    elif file_type == 'audio':
        return ext in ALLOWED_AUDIO_EXTENSIONS
    elif file_type == 'image':
        return ext in ALLOWED_IMAGE_EXTENSIONS
    return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    # Determine file type
    file_type = None
    if allowed_file(file.filename, 'video'):
        file_type = 'video'
    elif allowed_file(file.filename, 'audio'):
        file_type = 'audio'
    elif allowed_file(file.filename, 'image'):
        file_type = 'image'
    else:
        flash('Invalid file type. Please upload video, audio, or image files.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename or "unknown")
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Get file info
        file_size = os.path.getsize(file_path)
        
        # Start analysis
        start_time = time.time()
        
        # Initialize enhanced analyzers
        video_analyzer = EnhancedVideoAnalyzer()
        audio_analyzer = AudioAnalyzer()
        image_analyzer = EnhancedImageAnalyzer()
        ensemble_scorer = EnsembleScorer()
        
        analysis_results = {}
        
        if file_type == 'video':
            # Analyze video
            video_results = video_analyzer.analyze(file_path)
            analysis_results.update(video_results)
            
            # Try to extract and analyze audio from video
            try:
                audio_path = FileHandler.extract_audio_from_video(file_path)
                if audio_path:
                    audio_results = audio_analyzer.analyze(audio_path)
                    analysis_results.update(audio_results)
                    # Clean up temporary audio file
                    os.remove(audio_path)
            except Exception as e:
                app.logger.warning(f"Could not extract audio from video: {e}")
        
        elif file_type == 'audio':
            # Analyze audio only
            audio_results = audio_analyzer.analyze(file_path)
            analysis_results.update(audio_results)
        
        elif file_type == 'image':
            # Analyze image only
            image_results = image_analyzer.analyze(file_path)
            analysis_results.update(image_results)
        
        # Calculate ensemble score
        confidence_score = ensemble_scorer.calculate_score(analysis_results, file_type)
        
        processing_time = time.time() - start_time
        
        # Save analysis to database
        analysis = Analysis()
        analysis.filename = file.filename or "unknown"
        analysis.file_type = file_type
        analysis.file_path = file_path
        analysis.confidence_score = confidence_score
        analysis.video_score = analysis_results.get('video_score')
        analysis.audio_score = analysis_results.get('audio_score')
        analysis.image_score = analysis_results.get('image_score')
        analysis.face_consistency_score = analysis_results.get('face_consistency_score')
        analysis.frame_artifacts_score = analysis_results.get('frame_artifacts_score')
        analysis.spectral_anomaly_score = analysis_results.get('spectral_anomaly_score')
        analysis.frequency_pattern_score = analysis_results.get('frequency_pattern_score')
        analysis.face_analysis_score = analysis_results.get('face_analysis_score')
        analysis.compression_artifacts_score = analysis_results.get('compression_artifacts_score')
        analysis.pixel_inconsistency_score = analysis_results.get('pixel_inconsistency_score')
        analysis.metadata_score = analysis_results.get('metadata_score')
        analysis.faceforensics_score = analysis_results.get('faceforensics_score')
        analysis.file_size = file_size
        analysis.duration = analysis_results.get('duration')
        analysis.processing_time = processing_time
        analysis.analysis_details = json.dumps(analysis_results)
        
        db.session.add(analysis)
        db.session.commit()
        
        flash('Analysis completed successfully!', 'success')
        return redirect(url_for('results', analysis_id=analysis.id))
        
    except Exception as e:
        app.logger.error(f"Error during analysis: {e}")
        flash(f'Error processing file: {str(e)}', 'error')
        # Clean up file if it exists
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except (NameError, UnboundLocalError):
            pass  # file_path was never defined
        return redirect(url_for('index'))

@app.route('/results/<int:analysis_id>')
def results(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    
    # Parse analysis details
    details = {}
    if analysis.analysis_details:
        try:
            details = json.loads(analysis.analysis_details)
        except json.JSONDecodeError:
            details = {}
    
    return render_template('results.html', analysis=analysis, details=details)

@app.route('/history')
def history():
    page = request.args.get('page', 1, type=int)
    analyses = Analysis.query.order_by(Analysis.created_at.desc()).paginate(
        page=page, per_page=10, error_out=False
    )
    return render_template('history.html', analyses=analyses)

@app.route('/api/analysis/<int:analysis_id>')
def api_analysis(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    return jsonify(analysis.to_dict())

@app.errorhandler(413)
def too_large(e):
    flash('File too large. Maximum size is 100MB.', 'error')
    return redirect(url_for('index'))

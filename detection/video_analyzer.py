import cv2
import numpy as np
import os
from typing import Dict, List, Tuple
try:
    from detection.faceforensics_detector import FaceForensicsDetector
    FACEFORENSICS_AVAILABLE = True
except ImportError:
    FACEFORENSICS_AVAILABLE = False

class VideoAnalyzer:
    def __init__(self):
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize FaceForensics++ detector if available
        self.faceforensics_detector = None
        if FACEFORENSICS_AVAILABLE:
            try:
                self.faceforensics_detector = FaceForensicsDetector()
            except Exception as e:
                print(f"Could not initialize FaceForensics++ detector: {e}")
        
    def analyze(self, video_path: str) -> Dict:
        """
        Analyze video for deepfake indicators
        Returns analysis results as dictionary
        """
        results = {
            'video_score': 0.0,
            'face_consistency_score': 0.0,
            'frame_artifacts_score': 0.0,
            'faceforensics_score': 0.0,
            'duration': 0.0,
            'total_frames': 0,
            'faces_detected': 0,
            'consistency_violations': 0,
            'artifact_detections': 0
        }
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Could not open video file")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            results['duration'] = frame_count / fps if fps > 0 else 0
            results['total_frames'] = frame_count
            
            # Sample frames for analysis (every 10th frame to reduce processing time)
            sample_interval = max(1, frame_count // 100)  # Analyze max 100 frames
            
            face_data = []
            frame_artifacts = []
            faceforensics_scores = []
            prev_frame = None
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % sample_interval == 0:
                    # Analyze this frame
                    face_info = self._detect_faces(frame)
                    artifact_score = self._detect_artifacts(frame, prev_frame)
                    
                    # FaceForensics++ analysis
                    faceforensics_score = 0.0
                    if self.faceforensics_detector:
                        ff_result = self.faceforensics_detector.analyze_video_frame(frame)
                        faceforensics_score = ff_result.get('faceforensics_score', 0.0)
                    
                    face_data.append(face_info)
                    frame_artifacts.append(artifact_score)
                    faceforensics_scores.append(faceforensics_score)
                    
                    prev_frame = frame.copy()
                
                frame_idx += 1
            
            cap.release()
            
            # Calculate consistency scores
            results['face_consistency_score'] = self._calculate_face_consistency(face_data)
            results['frame_artifacts_score'] = np.mean(frame_artifacts) if frame_artifacts else 0
            results['faceforensics_score'] = np.mean(faceforensics_scores) if faceforensics_scores else 0
            results['faces_detected'] = sum(1 for fd in face_data if fd['face_count'] > 0)
            
            # Calculate overall video score (higher = more likely fake)
            results['video_score'] = self._calculate_video_score(results)
            
        except Exception as e:
            print(f"Error analyzing video: {e}")
            results['error'] = str(e)
        
        return results
    
    def _detect_faces(self, frame) -> Dict:
        """Detect faces in frame and return face information"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        face_info = {
            'face_count': len(faces),
            'face_areas': [],
            'face_positions': []
        }
        
        for (x, y, w, h) in faces:
            face_info['face_areas'].append(w * h)
            face_info['face_positions'].append((x + w//2, y + h//2))  # Center point
        
        return face_info
    
    def _detect_artifacts(self, frame, prev_frame) -> float:
        """Detect frame artifacts that might indicate deepfakes"""
        if prev_frame is None:
            return 0.0
        
        # Convert to grayscale
        gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate frame difference
        diff = cv2.absdiff(gray_current, gray_prev)
        
        # Look for unusual patterns
        # 1. Excessive noise in difference
        noise_level = np.std(diff)
        
        # 2. Unusual edge patterns
        edges_current = cv2.Canny(gray_current, 50, 150)
        edges_prev = cv2.Canny(gray_prev, 50, 150)
        edge_diff = cv2.absdiff(edges_current, edges_prev)
        edge_inconsistency = np.mean(edge_diff)
        
        # 3. Detect potential compression artifacts
        # Check for blocking artifacts (8x8 DCT blocks)
        h, w = gray_current.shape
        block_variance = []
        for y in range(0, h-8, 8):
            for x in range(0, w-8, 8):
                block = gray_current[y:y+8, x:x+8]
                block_variance.append(np.var(block))
        
        variance_inconsistency = np.std(block_variance) if block_variance else 0
        
        # Combine metrics (normalize and weight)
        artifact_score = (
            min(noise_level / 50.0, 1.0) * 0.4 +
            min(edge_inconsistency / 100.0, 1.0) * 0.4 +
            min(variance_inconsistency / 1000.0, 1.0) * 0.2
        )
        
        return artifact_score
    
    def _calculate_face_consistency(self, face_data: List[Dict]) -> float:
        """Calculate face consistency score across frames"""
        if not face_data:
            return 0.0
        
        face_counts = [fd['face_count'] for fd in face_data]
        
        # Check for consistent face count
        if len(set(face_counts)) > 1:
            count_inconsistency = 0.5
        else:
            count_inconsistency = 0.0
        
        # Check face size consistency (when faces are present)
        size_inconsistencies = []
        prev_areas = None
        
        for fd in face_data:
            if fd['face_count'] > 0 and fd['face_areas']:
                avg_area = np.mean(fd['face_areas'])
                if prev_areas is not None:
                    # Calculate relative change in face size
                    size_change = abs(avg_area - prev_areas) / max(prev_areas, 1)
                    size_inconsistencies.append(size_change)
                prev_areas = avg_area
        
        size_inconsistency = np.mean(size_inconsistencies) if size_inconsistencies else 0.0
        
        # Check position consistency
        position_inconsistencies = []
        prev_positions = None
        
        for fd in face_data:
            if fd['face_count'] > 0 and fd['face_positions']:
                avg_pos = np.mean(fd['face_positions'], axis=0) if len(fd['face_positions']) > 0 else [0, 0]
                if prev_positions is not None:
                    # Calculate position change
                    pos_change = np.linalg.norm(np.array(avg_pos) - np.array(prev_positions))
                    position_inconsistencies.append(pos_change)
                prev_positions = avg_pos
        
        position_inconsistency = min(np.mean(position_inconsistencies) / 100.0, 1.0) if position_inconsistencies else 0.0
        
        # Combine inconsistencies
        consistency_score = (count_inconsistency * 0.4 + 
                           min(size_inconsistency, 1.0) * 0.3 + 
                           position_inconsistency * 0.3)
        
        return consistency_score
    
    def _calculate_video_score(self, results: Dict) -> float:
        """Calculate overall video deepfake probability score"""
        face_score = results.get('face_consistency_score', 0)
        artifact_score = results.get('frame_artifacts_score', 0)
        faceforensics_score = results.get('faceforensics_score', 0)
        
        # Weight the scores - FaceForensics++ gets higher weight due to its proven accuracy
        if faceforensics_score > 0:
            video_score = (face_score * 0.3 + 
                          artifact_score * 0.2 + 
                          faceforensics_score / 100 * 0.5)  # Convert to 0-1 scale
        else:
            video_score = face_score * 0.6 + artifact_score * 0.4
        
        # Normalize to 0-100 scale
        return min(video_score * 100, 100)

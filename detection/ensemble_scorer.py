import numpy as np
from typing import Dict

class EnsembleScorer:
    def __init__(self):
        # Weights for different analysis types
        self.video_weight = 0.6
        self.audio_weight = 0.4
        self.image_weight = 0.5
        self.xception_weight = 0.7  # Xception gets higher weight due to proven accuracy
        
        # Thresholds for classification
        self.low_confidence_threshold = 30
        self.high_confidence_threshold = 70
    
    def calculate_score(self, analysis_results: Dict, file_type: str) -> float:
        """
        Calculate ensemble deepfake probability score
        Returns score from 0-100 (0 = likely real, 100 = likely fake)
        """
        scores = []
        weights = []
        
        # Video analysis scores
        if 'video_score' in analysis_results and analysis_results['video_score'] is not None:
            video_score = analysis_results['video_score']
            scores.append(video_score)
            weights.append(self.video_weight)
        
        # Audio analysis scores
        if 'audio_score' in analysis_results and analysis_results['audio_score'] is not None:
            audio_score = analysis_results['audio_score']
            scores.append(audio_score)
            weights.append(self.audio_weight)
        
        # Image analysis scores
        if 'image_score' in analysis_results and analysis_results['image_score'] is not None:
            image_score = analysis_results['image_score']
            scores.append(image_score)
            weights.append(self.image_weight)
        
        # Xception model scores (highest priority)
        if 'xception_score' in analysis_results and analysis_results['xception_score'] is not None:
            xception_score = analysis_results['xception_score']
            scores.append(xception_score)
            weights.append(self.xception_weight)
        
        if not scores:
            return 0.0  # No analysis performed
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        
        normalized_weights = [w / total_weight for w in weights]
        
        # Calculate weighted average
        ensemble_score = sum(score * weight for score, weight in zip(scores, normalized_weights))
        
        # Apply file-type specific adjustments
        ensemble_score = self._apply_file_type_adjustments(ensemble_score, file_type, analysis_results)
        
        # Apply confidence calibration
        ensemble_score = self._calibrate_confidence(ensemble_score, analysis_results)
        
        return min(max(ensemble_score, 0.0), 100.0)
    
    def _apply_file_type_adjustments(self, score: float, file_type: str, results: Dict) -> float:
        """Apply adjustments based on file type and quality indicators"""
        
        if file_type == 'video':
            # Video-specific adjustments
            
            # If very few faces detected, reduce confidence in face-based detection
            faces_detected = results.get('faces_detected', 0)
            total_frames = results.get('total_frames', 1)
            face_detection_ratio = faces_detected / max(total_frames / 10, 1)  # Sampled frames
            
            if face_detection_ratio < 0.1:  # Very few faces detected
                score *= 0.8  # Reduce confidence
            
            # Adjust based on video duration
            duration = results.get('duration', 0)
            if duration < 2:  # Very short videos are harder to analyze
                score *= 0.9
            elif duration > 60:  # Very long videos might have more artifacts
                score *= 1.1
        
        elif file_type == 'audio':
            # Audio-specific adjustments
            
            # Adjust based on audio duration
            duration = results.get('duration', 0)
            if duration < 1:  # Very short audio clips
                score *= 0.8
            elif duration > 30:  # Longer audio might have more patterns
                score *= 1.05
            
            # Adjust based on sample rate
            sample_rate = results.get('sample_rate', 22050)
            if sample_rate < 16000:  # Low quality audio
                score *= 0.9
        
        elif file_type == 'image':
            # Image-specific adjustments
            
            # Adjust based on image dimensions
            width = results.get('image_width', 0)
            height = results.get('image_height', 0)
            
            if width < 100 or height < 100:  # Very small images
                score *= 0.7
            elif width > 4000 or height > 4000:  # Very large images
                score *= 1.1
            
            # Adjust based on faces detected
            faces_detected = results.get('faces_detected', 0)
            if faces_detected == 0:  # No faces detected
                score *= 0.6  # Reduce confidence for face-based detection
        
        return score
    
    def _calibrate_confidence(self, score: float, results: Dict) -> float:
        """Calibrate confidence based on analysis quality and consistency"""
        
        # Check for consistency between different analysis methods
        consistency_bonus = self._calculate_consistency_bonus(results)
        score += consistency_bonus
        
        # Apply quality-based adjustments
        quality_adjustment = self._calculate_quality_adjustment(results)
        score *= quality_adjustment
        
        return score
    
    def _calculate_consistency_bonus(self, results: Dict) -> float:
        """Calculate bonus/penalty based on consistency between analysis methods"""
        
        video_score = results.get('video_score', 0)
        audio_score = results.get('audio_score', 0)
        image_score = results.get('image_score', 0)
        
        # Check consistency between different analysis methods
        scores = [s for s in [video_score, audio_score, image_score] if s > 0]
        
        if len(scores) >= 2:
            # Check if they agree in their assessment
            max_diff = max(scores) - min(scores)
            
            if max_diff < 20:  # Good agreement
                return 5  # Boost confidence
            elif max_diff > 50:  # Poor agreement
                return -10  # Reduce confidence
        
        return 0  # No adjustment
    
    def _calculate_quality_adjustment(self, results: Dict) -> float:
        """Calculate quality adjustment factor"""
        
        adjustment_factor = 1.0
        
        # File size considerations
        file_size = results.get('file_size', 0)
        if file_size < 100000:  # Very small files (< 100KB)
            adjustment_factor *= 0.9
        
        # Duration considerations
        duration = results.get('duration', 0)
        if duration < 0.5:  # Very short duration
            adjustment_factor *= 0.85
        
        # Processing time considerations (very fast processing might indicate errors)
        processing_time = results.get('processing_time', 1)
        if processing_time < 0.1:  # Suspiciously fast
            adjustment_factor *= 0.8
        
        return adjustment_factor
    
    def get_confidence_level(self, score: float) -> str:
        """Get human-readable confidence level"""
        if score < self.low_confidence_threshold:
            return "Low"
        elif score < self.high_confidence_threshold:
            return "Medium"
        else:
            return "High"
    
    def get_verdict(self, score: float) -> str:
        """Get human-readable verdict"""
        if score < 20:
            return "Likely Real"
        elif score < 40:
            return "Probably Real"
        elif score < 60:
            return "Uncertain"
        elif score < 80:
            return "Probably Fake"
        else:
            return "Likely Fake"
    
    def get_risk_level(self, score: float) -> str:
        """Get risk level classification"""
        if score < 30:
            return "low"
        elif score < 70:
            return "medium"
        else:
            return "high"

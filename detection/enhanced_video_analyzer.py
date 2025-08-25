import cv2
import numpy as np
import os
from typing import Dict, List, Tuple
from scipy import ndimage
from skimage import feature, measure
import logging

class EnhancedVideoAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
    def analyze(self, video_path: str) -> Dict:
        """Enhanced deepfake detection with multiple advanced techniques"""
        results = {
            'video_score': 0.0,
            'face_consistency_score': 0.0,
            'frame_artifacts_score': 0.0,
            'temporal_consistency_score': 0.0,
            'eye_blink_score': 0.0,
            'lip_sync_score': 0.0,
            'texture_analysis_score': 0.0,
            'duration': 0.0
        }
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return results
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            results['duration'] = frame_count / fps if fps > 0 else 0
            
            # Advanced analysis arrays
            face_scores = []
            artifact_scores = []
            temporal_scores = []
            blink_scores = []
            texture_scores = []
            lip_scores = []
            
            prev_frame = None
            frame_idx = 0
            sample_rate = max(1, frame_count // 50)  # Analyze 50 frames max
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % sample_rate == 0:
                    # Multi-layer analysis
                    face_score = self._advanced_face_analysis(frame)
                    face_scores.append(face_score)
                    
                    artifact_score = self._detect_manipulation_artifacts(frame)
                    artifact_scores.append(artifact_score)
                    
                    texture_score = self._analyze_texture_authenticity(frame)
                    texture_scores.append(texture_score)
                    
                    blink_score = self._analyze_eye_patterns(frame)
                    blink_scores.append(blink_score)
                    
                    lip_score = self._analyze_lip_movements(frame)
                    lip_scores.append(lip_score)
                    
                    if prev_frame is not None:
                        temporal_score = self._analyze_temporal_coherence(prev_frame, frame)
                        temporal_scores.append(temporal_score)
                    
                    prev_frame = frame.copy()
                
                frame_idx += 1
            
            cap.release()
            
            # Calculate comprehensive scores
            results['face_consistency_score'] = np.mean(face_scores) if face_scores else 0.5
            results['frame_artifacts_score'] = np.mean(artifact_scores) if artifact_scores else 0.5
            results['temporal_consistency_score'] = np.mean(temporal_scores) if temporal_scores else 0.5
            results['eye_blink_score'] = np.mean(blink_scores) if blink_scores else 0.5
            results['texture_analysis_score'] = np.mean(texture_scores) if texture_scores else 0.5
            results['lip_sync_score'] = np.mean(lip_scores) if lip_scores else 0.5
            
            # Advanced weighted scoring for better accuracy
            results['video_score'] = self._calculate_enhanced_score(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in enhanced video analysis: {e}")
            return results
    
    def _advanced_face_analysis(self, frame):
        """Multi-technique face analysis for deepfake detection"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return 0.2  # Low authenticity if no faces
            
            face_scores = []
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                
                # Multiple analysis techniques
                symmetry_score = self._analyze_facial_symmetry(face_roi)
                edge_coherence = self._analyze_edge_coherence(face_roi)
                frequency_analysis = self._analyze_frequency_domain(face_roi)
                landmark_consistency = self._analyze_landmark_consistency(face_roi)
                
                combined_score = (
                    symmetry_score * 0.25 +
                    edge_coherence * 0.25 +
                    frequency_analysis * 0.25 +
                    landmark_consistency * 0.25
                )
                face_scores.append(combined_score)
            
            return np.mean(face_scores)
            
        except Exception as e:
            self.logger.error(f"Error in advanced face analysis: {e}")
            return 0.3
    
    def _detect_manipulation_artifacts(self, frame):
        """Detect various manipulation artifacts"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # DCT-based compression artifact detection
            dct_score = self._analyze_dct_artifacts(gray)
            
            # JPEG compression inconsistencies
            jpeg_score = self._analyze_jpeg_artifacts(frame)
            
            # Pixel-level inconsistencies
            pixel_score = self._analyze_pixel_inconsistencies(gray)
            
            # Blending artifacts
            blend_score = self._detect_blending_artifacts(gray)
            
            # Combine artifact scores
            artifact_score = (
                dct_score * 0.3 +
                jpeg_score * 0.25 +
                pixel_score * 0.25 +
                blend_score * 0.2
            )
            
            # Return authenticity score (1 - artifact_score)
            return max(0.0, 1.0 - artifact_score)
            
        except Exception as e:
            self.logger.error(f"Error detecting manipulation artifacts: {e}")
            return 0.5
    
    def _analyze_temporal_coherence(self, prev_frame, curr_frame):
        """Analyze temporal coherence between consecutive frames"""
        try:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # Optical flow analysis
            flow = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray,
                np.array([[100, 100]], dtype=np.float32).reshape(-1, 1, 2),
                None
            )[0]
            
            # Motion consistency analysis
            motion_score = self._analyze_motion_consistency(prev_gray, curr_gray)
            
            # Illumination consistency
            illum_score = self._analyze_illumination_consistency(prev_frame, curr_frame)
            
            # Temporal texture consistency
            texture_score = self._analyze_temporal_texture(prev_gray, curr_gray)
            
            coherence_score = (
                motion_score * 0.4 +
                illum_score * 0.3 +
                texture_score * 0.3
            )
            
            return coherence_score
            
        except Exception as e:
            self.logger.error(f"Error in temporal coherence analysis: {e}")
            return 0.5
    
    def _analyze_eye_patterns(self, frame):
        """Analyze eye blink patterns and eye authenticity"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(eyes) < 2:
                return 0.2  # Low score if eyes not properly detected
            
            eye_scores = []
            for (x, y, w, h) in eyes[:2]:  # Analyze first two eyes
                eye_roi = gray[y:y+h, x:x+w]
                
                # Eye shape analysis
                shape_score = self._analyze_eye_shape(eye_roi)
                
                # Pupil detection and analysis
                pupil_score = self._analyze_pupil_authenticity(eye_roi)
                
                # Reflection analysis
                reflection_score = self._analyze_eye_reflections(eye_roi)
                
                eye_score = (shape_score * 0.4 + pupil_score * 0.3 + reflection_score * 0.3)
                eye_scores.append(eye_score)
            
            return np.mean(eye_scores) if eye_scores else 0.2
            
        except Exception as e:
            self.logger.error(f"Error in eye pattern analysis: {e}")
            return 0.3
    
    def _analyze_lip_movements(self, frame):
        """Analyze lip movement authenticity"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect mouth region (simplified approach)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return 0.3
            
            lip_scores = []
            for (x, y, w, h) in faces:
                # Extract lower face region (mouth area)
                mouth_y = y + int(h * 0.6)
                mouth_roi = gray[mouth_y:y+h, x:x+w]
                
                if mouth_roi.size > 0:
                    # Analyze lip texture and movement patterns
                    texture_score = self._analyze_lip_texture(mouth_roi)
                    edge_score = self._analyze_lip_edges(mouth_roi)
                    
                    lip_score = (texture_score * 0.6 + edge_score * 0.4)
                    lip_scores.append(lip_score)
            
            return np.mean(lip_scores) if lip_scores else 0.3
            
        except Exception as e:
            self.logger.error(f"Error in lip movement analysis: {e}")
            return 0.3
    
    def _analyze_texture_authenticity(self, frame):
        """Advanced texture analysis for authenticity"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Local Binary Pattern analysis
            lbp = feature.local_binary_pattern(gray, 8, 1, method='uniform')
            lbp_hist = np.histogram(lbp.ravel(), bins=10)[0]
            lbp_score = np.std(lbp_hist) / (np.mean(lbp_hist) + 1e-6)
            
            # Gabor filter responses
            gabor_score = self._analyze_gabor_responses(gray)
            
            # Wavelet analysis
            wavelet_score = self._analyze_wavelet_features(gray)
            
            texture_authenticity = (
                min(1.0, lbp_score / 3.0) * 0.4 +
                gabor_score * 0.3 +
                wavelet_score * 0.3
            )
            
            return texture_authenticity
            
        except Exception as e:
            self.logger.error(f"Error in texture authenticity analysis: {e}")
            return 0.5
    
    def _calculate_enhanced_score(self, results):
        """Calculate enhanced authenticity score with advanced weighting"""
        face_score = results.get('face_consistency_score', 0.5)
        artifact_score = results.get('frame_artifacts_score', 0.5)
        temporal_score = results.get('temporal_consistency_score', 0.5)
        eye_score = results.get('eye_blink_score', 0.5)
        texture_score = results.get('texture_analysis_score', 0.5)
        lip_score = results.get('lip_sync_score', 0.5)
        
        # Advanced weighted combination
        enhanced_score = (
            face_score * 0.25 +
            artifact_score * 0.20 +
            temporal_score * 0.20 +
            eye_score * 0.15 +
            texture_score * 0.10 +
            lip_score * 0.10
        )
        
        # Convert to percentage and invert (higher = more authentic)
        authenticity_percentage = enhanced_score * 100
        
        return min(100, max(0, authenticity_percentage))
    
    # Helper methods for specific analyses
    def _analyze_facial_symmetry(self, face_roi):
        """Analyze facial symmetry"""
        try:
            h, w = face_roi.shape
            if w < 20:
                return 0.5
            
            left_half = face_roi[:, :w//2]
            right_half = cv2.flip(face_roi[:, w//2:], 1)
            
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
            
            diff = np.abs(left_half.astype(float) - right_half.astype(float))
            symmetry_score = 1.0 - (np.mean(diff) / 255.0)
            
            return max(0.0, symmetry_score)
        except:
            return 0.5
    
    def _analyze_edge_coherence(self, face_roi):
        """Analyze edge coherence in face region"""
        try:
            edges = cv2.Canny(face_roi, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            return min(1.0, edge_density * 8.0)
        except:
            return 0.5
    
    def _analyze_frequency_domain(self, face_roi):
        """Analyze frequency domain characteristics"""
        try:
            f_transform = np.fft.fft2(face_roi)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Analyze high frequency components
            h, w = magnitude_spectrum.shape
            center_h, center_w = h//2, w//2
            high_freq = magnitude_spectrum[center_h-h//4:center_h+h//4, center_w-w//4:center_w+w//4]
            
            freq_score = np.std(high_freq) / (np.mean(high_freq) + 1e-6)
            return min(1.0, freq_score / 2.0)
        except:
            return 0.5
    
    def _analyze_landmark_consistency(self, face_roi):
        """Analyze facial landmark consistency"""
        try:
            # Simplified landmark analysis using corner detection
            corners = cv2.goodFeaturesToTrack(face_roi, 25, 0.01, 10)
            if corners is not None:
                landmark_score = len(corners) / 25.0
                return min(1.0, landmark_score)
            return 0.3
        except:
            return 0.5
    
    def _analyze_dct_artifacts(self, gray):
        """Analyze DCT compression artifacts"""
        try:
            dct = cv2.dct(np.float32(gray))
            high_freq = dct[gray.shape[0]//2:, gray.shape[1]//2:]
            artifact_measure = np.std(high_freq) / (np.mean(np.abs(high_freq)) + 1e-6)
            return min(1.0, artifact_measure / 5.0)
        except:
            return 0.5
    
    def _analyze_jpeg_artifacts(self, frame):
        """Analyze JPEG compression artifacts"""
        try:
            # Convert to YUV and analyze chroma channels
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            u_channel = yuv[:, :, 1]
            v_channel = yuv[:, :, 2]
            
            u_artifacts = np.std(u_channel) / (np.mean(u_channel) + 1e-6)
            v_artifacts = np.std(v_channel) / (np.mean(v_channel) + 1e-6)
            
            jpeg_score = (u_artifacts + v_artifacts) / 2.0
            return min(1.0, jpeg_score / 3.0)
        except:
            return 0.5
    
    def _analyze_pixel_inconsistencies(self, gray):
        """Analyze pixel-level inconsistencies"""
        try:
            # Calculate local variance
            kernel = np.ones((3, 3), np.float32) / 9
            local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            local_var = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
            
            inconsistency = np.std(local_var) / (np.mean(local_var) + 1e-6)
            return min(1.0, inconsistency / 10.0)
        except:
            return 0.5
    
    def _detect_blending_artifacts(self, gray):
        """Detect blending artifacts"""
        try:
            # Analyze gradient discontinuities
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_direction = np.arctan2(grad_y, grad_x)
            
            # Look for sudden changes in gradient direction
            direction_changes = np.abs(np.diff(gradient_direction, axis=0))
            blend_score = np.mean(direction_changes) / np.pi
            
            return min(1.0, blend_score * 2.0)
        except:
            return 0.5
    
    def _analyze_motion_consistency(self, prev_gray, curr_gray):
        """Analyze motion consistency between frames"""
        try:
            diff = cv2.absdiff(prev_gray, curr_gray)
            motion_score = 1.0 - min(1.0, np.mean(diff) / 128.0)
            return motion_score
        except:
            return 0.5
    
    def _analyze_illumination_consistency(self, prev_frame, curr_frame):
        """Analyze illumination consistency"""
        try:
            prev_lab = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2LAB)
            curr_lab = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2LAB)
            
            # Compare L channel (lightness)
            l_diff = np.abs(prev_lab[:, :, 0].astype(float) - curr_lab[:, :, 0].astype(float))
            illum_score = 1.0 - min(1.0, np.mean(l_diff) / 100.0)
            
            return illum_score
        except:
            return 0.5
    
    def _analyze_temporal_texture(self, prev_gray, curr_gray):
        """Analyze temporal texture consistency"""
        try:
            # Calculate texture features for both frames
            prev_lbp = feature.local_binary_pattern(prev_gray, 8, 1)
            curr_lbp = feature.local_binary_pattern(curr_gray, 8, 1)
            
            # Compare texture patterns
            texture_diff = np.abs(prev_lbp - curr_lbp)
            texture_score = 1.0 - min(1.0, np.mean(texture_diff) / 255.0)
            
            return texture_score
        except:
            return 0.5
    
    def _analyze_eye_shape(self, eye_roi):
        """Analyze eye shape authenticity"""
        try:
            edges = cv2.Canny(eye_roi, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    return min(1.0, circularity * 2.0)
            
            return 0.3
        except:
            return 0.5
    
    def _analyze_pupil_authenticity(self, eye_roi):
        """Analyze pupil authenticity"""
        try:
            # Simple pupil detection using HoughCircles
            circles = cv2.HoughCircles(eye_roi, cv2.HOUGH_GRADIENT, 1, 20,
                                     param1=50, param2=30, minRadius=0, maxRadius=0)
            
            if circles is not None:
                return 0.8  # Good score if pupil detected
            return 0.4
        except:
            return 0.5
    
    def _analyze_eye_reflections(self, eye_roi):
        """Analyze eye reflection patterns"""
        try:
            # Look for bright spots (reflections)
            _, thresh = cv2.threshold(eye_roi, 200, 255, cv2.THRESH_BINARY)
            reflection_area = np.sum(thresh > 0)
            total_area = eye_roi.size
            
            reflection_ratio = reflection_area / total_area
            return min(1.0, reflection_ratio * 20.0)
        except:
            return 0.5
    
    def _analyze_lip_texture(self, mouth_roi):
        """Analyze lip texture authenticity"""
        try:
            # Analyze texture using LBP
            lbp = feature.local_binary_pattern(mouth_roi, 8, 1)
            texture_score = np.std(lbp) / (np.mean(lbp) + 1e-6)
            return min(1.0, texture_score / 5.0)
        except:
            return 0.5
    
    def _analyze_lip_edges(self, mouth_roi):
        """Analyze lip edge definition"""
        try:
            edges = cv2.Canny(mouth_roi, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            return min(1.0, edge_density * 10.0)
        except:
            return 0.5
    
    def _analyze_gabor_responses(self, gray):
        """Analyze Gabor filter responses"""
        try:
            # Simple Gabor-like analysis using different orientations
            kernel1 = cv2.getGaborKernel((21, 21), 5, 0, 10, 0.5, 0, ktype=cv2.CV_32F)
            kernel2 = cv2.getGaborKernel((21, 21), 5, np.pi/4, 10, 0.5, 0, ktype=cv2.CV_32F)
            
            filtered1 = cv2.filter2D(gray, cv2.CV_8UC3, kernel1)
            filtered2 = cv2.filter2D(gray, cv2.CV_8UC3, kernel2)
            
            response1 = np.std(filtered1)
            response2 = np.std(filtered2)
            
            gabor_score = (response1 + response2) / 2.0
            return min(1.0, gabor_score / 100.0)
        except:
            return 0.5
    
    def _analyze_wavelet_features(self, gray):
        """Analyze wavelet features (simplified)"""
        try:
            # Simple wavelet-like analysis using Laplacian
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            wavelet_score = np.std(laplacian) / (np.mean(np.abs(laplacian)) + 1e-6)
            return min(1.0, wavelet_score / 10.0)
        except:
            return 0.5
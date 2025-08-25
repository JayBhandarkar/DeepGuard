import cv2
import numpy as np
import os
from typing import Dict, List, Tuple
try:
    from detection.faceforensics_detector import FaceForensicsDetector
    FACEFORENSICS_AVAILABLE = True
except ImportError:
    FACEFORENSICS_AVAILABLE = False

class ImageAnalyzer:
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
        
    def analyze(self, image_path: str) -> Dict:
        """
        Analyze image for deepfake indicators
        Returns analysis results as dictionary
        """
        results = {
            'image_score': 0.0,
            'face_analysis_score': 0.0,
            'compression_artifacts_score': 0.0,
            'pixel_inconsistency_score': 0.0,
            'metadata_score': 0.0,
            'faceforensics_score': 0.0,
            'faces_detected': 0,
            'image_width': 0,
            'image_height': 0,
            'file_size': 0
        }
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise Exception("Could not load image file")
            
            results['image_height'], results['image_width'] = image.shape[:2]
            results['file_size'] = os.path.getsize(image_path)
            
            # Store reference to results for face detection callback
            self._current_results = results
            
            # Perform face analysis
            face_score = self._analyze_faces(image)
            results['face_analysis_score'] = face_score
            
            # FaceForensics++ analysis
            if self.faceforensics_detector:
                ff_result = self.faceforensics_detector.analyze_image(image_path)
                results['faceforensics_score'] = ff_result.get('faceforensics_score', 0.0)
                # Update faces detected count if FaceForensics++ found more faces
                results['faces_detected'] = max(results['faces_detected'], ff_result.get('faces_detected', 0))
            
            # Analyze compression artifacts
            compression_score = self._analyze_compression_artifacts(image)
            results['compression_artifacts_score'] = compression_score
            
            # Analyze pixel inconsistencies
            pixel_score = self._analyze_pixel_inconsistencies(image)
            results['pixel_inconsistency_score'] = pixel_score
            
            # Analyze metadata (basic)
            metadata_score = self._analyze_metadata(image_path)
            results['metadata_score'] = metadata_score
            
            # Calculate overall image score
            results['image_score'] = self._calculate_image_score(results)
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            results['error'] = str(e)
        
        return results
    
    def _analyze_faces(self, image) -> float:
        """Analyze faces for deepfake indicators"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Update faces detected count
            if hasattr(self, '_current_results'):
                self._current_results['faces_detected'] = len(faces)
            
            if len(faces) == 0:
                return 0.0  # No faces detected
            
            face_scores = []
            
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = image[y:y+h, x:x+w]
                
                # Analyze face for inconsistencies
                face_score = self._analyze_single_face(face_roi)
                face_scores.append(face_score)
            
            return np.mean(face_scores) if face_scores else 0.0
            
        except Exception as e:
            print(f"Error in face analysis: {e}")
            return 0.0
    
    def _analyze_single_face(self, face_roi) -> float:
        """Analyze a single face for deepfake indicators"""
        try:
            # Convert to different color spaces for analysis
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # 1. Analyze edge consistency
            edges = cv2.Canny(gray_face, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # 2. Analyze texture patterns
            # Look for unnatural smoothness or artificial patterns
            texture_variance = np.var(gray_face)
            
            # 3. Analyze color distribution
            # Check for unusual color patterns in face
            b, g, r = cv2.split(face_roi)
            color_variance = np.var([np.mean(b), np.mean(g), np.mean(r)])
            
            # 4. Look for compression artifacts in face region
            # Check for blocking artifacts
            h, w = gray_face.shape
            block_variances = []
            for y in range(0, h-8, 8):
                for x in range(0, w-8, 8):
                    block = gray_face[y:y+8, x:x+8]
                    block_variances.append(np.var(block))
            
            blocking_inconsistency = np.std(block_variances) if block_variances else 0
            
            # Normalize and combine scores
            edge_score = min(edge_density * 10, 1.0)
            texture_score = min(texture_variance / 1000, 1.0)
            color_score = min(color_variance / 100, 1.0)
            blocking_score = min(blocking_inconsistency / 500, 1.0)
            
            face_anomaly_score = (edge_score * 0.3 + 
                                 texture_score * 0.25 + 
                                 color_score * 0.25 + 
                                 blocking_score * 0.2)
            
            return face_anomaly_score
            
        except Exception as e:
            print(f"Error analyzing single face: {e}")
            return 0.0
    
    def _analyze_compression_artifacts(self, image) -> float:
        """Analyze compression artifacts that might indicate manipulation"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # 1. Look for JPEG compression artifacts (8x8 DCT blocks)
            block_variances = []
            for y in range(0, h-8, 8):
                for x in range(0, w-8, 8):
                    block = gray[y:y+8, x:x+8]
                    block_variances.append(np.var(block))
            
            # Unnatural compression patterns
            if block_variances:
                variance_inconsistency = np.std(block_variances)
                # Look for too uniform or too varied compression
                compression_anomaly = min(variance_inconsistency / 1000, 1.0)
            else:
                compression_anomaly = 0.0
            
            # 2. Check for unusual frequency patterns
            # Apply DCT to detect manipulation
            dct = cv2.dct(np.float32(gray))
            
            # High frequency content analysis
            high_freq_mask = np.zeros_like(dct)
            h_dct, w_dct = dct.shape
            high_freq_mask[h_dct//2:, w_dct//2:] = 1
            
            high_freq_energy = np.sum(np.abs(dct * high_freq_mask))
            total_energy = np.sum(np.abs(dct))
            high_freq_ratio = high_freq_energy / (total_energy + 1e-10)
            
            # 3. Check for resizing artifacts
            # Look for aliasing patterns
            fft = np.fft.fft2(gray)
            fft_magnitude = np.abs(fft)
            fft_variance = np.var(fft_magnitude)
            
            # Normalize scores
            freq_score = min(high_freq_ratio * 5, 1.0)
            fft_score = min(fft_variance / 1000000, 1.0)
            
            compression_score = (compression_anomaly * 0.5 + 
                               freq_score * 0.3 + 
                               fft_score * 0.2)
            
            return compression_score
            
        except Exception as e:
            print(f"Error in compression analysis: {e}")
            return 0.0
    
    def _analyze_pixel_inconsistencies(self, image) -> float:
        """Analyze pixel-level inconsistencies"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 1. Analyze noise patterns
            # Apply Gaussian blur and check difference
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = cv2.absdiff(gray, blurred)
            noise_level = np.mean(noise)
            
            # 2. Check for unnatural gradients
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_variance = np.var(gradient_magnitude)
            
            # 3. Analyze local standard deviation
            # Look for unnatural smoothness or roughness
            kernel = np.ones((5, 5), np.float32) / 25
            local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
            variance_map = np.sqrt(local_variance)
            variance_inconsistency = np.std(variance_map)
            
            # 4. Check for copy-paste artifacts
            # Simple template matching for duplicate regions
            template_size = min(32, gray.shape[0]//4, gray.shape[1]//4)
            if template_size > 8:
                template = gray[:template_size, :template_size]
                match_result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                max_matches = np.sum(match_result > 0.8)  # High similarity threshold
                copy_paste_score = min(max_matches / 10, 1.0)
            else:
                copy_paste_score = 0.0
            
            # Normalize and combine scores
            noise_score = min(noise_level / 20, 1.0)
            gradient_score = min(gradient_variance / 10000, 1.0)
            variance_score = min(variance_inconsistency / 50, 1.0)
            
            pixel_inconsistency = (noise_score * 0.3 + 
                                 gradient_score * 0.3 + 
                                 variance_score * 0.2 + 
                                 copy_paste_score * 0.2)
            
            return pixel_inconsistency
            
        except Exception as e:
            print(f"Error in pixel analysis: {e}")
            return 0.0
    
    def _analyze_metadata(self, image_path: str) -> float:
        """Analyze image metadata for manipulation indicators"""
        try:
            # Basic metadata analysis
            # For more sophisticated metadata analysis, would use libraries like exifread
            
            file_size = os.path.getsize(image_path)
            
            # Load image to check dimensions
            image = cv2.imread(image_path)
            if image is None:
                return 0.0
                
            height, width = image.shape[:2]
            
            # Calculate expected file size ratio
            # Compressed images should have reasonable size ratios
            pixel_count = height * width * 3  # 3 channels
            compression_ratio = file_size / pixel_count
            
            # Unusual compression ratios might indicate manipulation
            if compression_ratio < 0.01 or compression_ratio > 0.5:
                metadata_anomaly = 0.3
            else:
                metadata_anomaly = 0.0
            
            # Check for unusual dimensions (common in generated images)
            aspect_ratio = width / height
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                metadata_anomaly += 0.2
            
            # Check for power-of-2 dimensions (common in generated images)
            if (width & (width - 1)) == 0 or (height & (height - 1)) == 0:
                metadata_anomaly += 0.1
            
            return min(metadata_anomaly, 1.0)
            
        except Exception as e:
            print(f"Error in metadata analysis: {e}")
            return 0.0
    
    def _calculate_image_score(self, results: Dict) -> float:
        """Calculate overall image deepfake probability score"""
        face_score = results.get('face_analysis_score', 0)
        compression_score = results.get('compression_artifacts_score', 0)
        pixel_score = results.get('pixel_inconsistency_score', 0)
        metadata_score = results.get('metadata_score', 0)
        faceforensics_score = results.get('faceforensics_score', 0)
        
        # Weight the scores based on reliability - FaceForensics++ gets higher weight
        if faceforensics_score > 0:
            image_score = (face_score * 0.2 + 
                          compression_score * 0.2 + 
                          pixel_score * 0.1 + 
                          metadata_score * 0.1 +
                          faceforensics_score / 100 * 0.4)  # Convert to 0-1 scale
        else:
            image_score = (face_score * 0.4 + 
                          compression_score * 0.3 + 
                          pixel_score * 0.2 + 
                          metadata_score * 0.1)
        
        # Normalize to 0-100 scale
        return min(image_score * 100, 100)
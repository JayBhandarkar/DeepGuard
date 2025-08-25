import cv2
import numpy as np
from typing import Dict
from scipy import ndimage
from skimage import feature, measure, filters
import logging

class EnhancedImageAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize Xception detector if available
        try:
            from .xception_detector import XceptionDetector
            self.xception_detector = XceptionDetector()
            self.use_xception = True
            self.logger.info("Xception detector initialized successfully")
        except ImportError as e:
            self.use_xception = False
            self.logger.warning(f"Xception detector not available: {e}")
        except Exception as e:
            self.use_xception = False
            self.logger.warning(f"Failed to initialize Xception detector: {e}")
        
    def analyze(self, image_path: str) -> Dict:
        """Enhanced deepfake detection for images"""
        results = {
            'image_score': 0.0,
            'face_analysis_score': 0.0,
            'compression_artifacts_score': 0.0,
            'pixel_inconsistency_score': 0.0,
            'metadata_score': 0.0,
            'frequency_analysis_score': 0.0,
            'edge_analysis_score': 0.0
        }
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return results
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Advanced multi-layer analysis
            face_score = self._advanced_face_analysis(image, gray)
            compression_score = self._detect_compression_artifacts(image)
            pixel_score = self._analyze_pixel_inconsistencies(gray)
            metadata_score = self._analyze_metadata_authenticity(image_path)
            frequency_score = self._analyze_frequency_domain(gray)
            edge_score = self._analyze_edge_authenticity(gray)
            
            # Xception-based deepfake detection (if available)
            xception_score = 0.0
            if self.use_xception:
                try:
                    xception_result = self.xception_detector.analyze_image(image_path)
                    xception_score = xception_result.get('xception_score', 0.0)
                    results['xception_score'] = xception_score
                    results['xception_details'] = xception_result
                    self.logger.info(f"Xception analysis completed: {xception_score:.2f}% fake probability")
                except Exception as e:
                    self.logger.error(f"Xception analysis failed: {e}")
                    results['xception_score'] = 0.0
                    results['xception_details'] = {'error': str(e)}
            else:
                results['xception_score'] = 0.0
                results['xception_details'] = {'status': 'not_available'}
            
            # Store individual scores
            results['face_analysis_score'] = face_score
            results['compression_artifacts_score'] = compression_score
            results['pixel_inconsistency_score'] = pixel_score
            results['metadata_score'] = metadata_score
            results['frequency_analysis_score'] = frequency_score
            results['edge_analysis_score'] = edge_score
            
            # Calculate overall authenticity score
            results['image_score'] = self._calculate_overall_score(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in enhanced image analysis: {e}")
            return results
    
    def _advanced_face_analysis(self, image, gray):
        """Advanced face analysis for deepfake detection"""
        try:
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return 0.3  # Lower score if no faces detected
            
            face_scores = []
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                face_color = image[y:y+h, x:x+w]
                
                # Multiple face authenticity checks
                symmetry_score = self._analyze_facial_symmetry(face_roi)
                skin_score = self._analyze_skin_texture(face_color)
                landmark_score = self._analyze_facial_landmarks(face_roi)
                lighting_score = self._analyze_face_lighting(face_roi)
                detail_score = self._analyze_facial_details(face_roi)
                
                combined_score = (
                    symmetry_score * 0.25 +
                    skin_score * 0.25 +
                    landmark_score * 0.20 +
                    lighting_score * 0.15 +
                    detail_score * 0.15
                )
                face_scores.append(combined_score)
            
            return np.mean(face_scores)
            
        except Exception as e:
            self.logger.error(f"Error in advanced face analysis: {e}")
            return 0.4
    
    def _detect_compression_artifacts(self, image):
        """Detect various compression artifacts"""
        try:
            # JPEG compression analysis
            jpeg_score = self._analyze_jpeg_compression(image)
            
            # DCT block artifacts
            dct_score = self._analyze_dct_blocks(image)
            
            # Quantization artifacts
            quant_score = self._analyze_quantization_artifacts(image)
            
            # Combine scores (higher = more authentic)
            compression_authenticity = (
                jpeg_score * 0.4 +
                dct_score * 0.3 +
                quant_score * 0.3
            )
            
            return compression_authenticity
            
        except Exception as e:
            self.logger.error(f"Error detecting compression artifacts: {e}")
            return 0.5
    
    def _analyze_pixel_inconsistencies(self, gray):
        """Analyze pixel-level inconsistencies"""
        try:
            # Local variance analysis
            local_var_score = self._analyze_local_variance(gray)
            
            # Noise pattern analysis
            noise_score = self._analyze_noise_patterns(gray)
            
            # Gradient consistency
            gradient_score = self._analyze_gradient_consistency(gray)
            
            # Statistical anomalies
            statistical_score = self._analyze_statistical_anomalies(gray)
            
            pixel_authenticity = (
                local_var_score * 0.3 +
                noise_score * 0.25 +
                gradient_score * 0.25 +
                statistical_score * 0.2
            )
            
            return pixel_authenticity
            
        except Exception as e:
            self.logger.error(f"Error in pixel inconsistency analysis: {e}")
            return 0.5
    
    def _analyze_metadata_authenticity(self, image_path):
        """Analyze image metadata for authenticity indicators"""
        try:
            # Basic metadata analysis (simplified)
            import os
            from PIL import Image
            from PIL.ExifTags import TAGS
            
            try:
                pil_image = Image.open(image_path)
                exifdata = pil_image.getexif()
                
                metadata_indicators = 0
                total_checks = 0
                
                # Check for camera information
                if exifdata:
                    for tag_id in exifdata:
                        tag = TAGS.get(tag_id, tag_id)
                        if tag in ['Make', 'Model', 'DateTime', 'Software']:
                            metadata_indicators += 1
                        total_checks += 1
                
                if total_checks > 0:
                    metadata_score = metadata_indicators / total_checks
                else:
                    metadata_score = 0.3  # No metadata might indicate manipulation
                
                return min(1.0, metadata_score * 2.0)
                
            except:
                return 0.3
                
        except Exception as e:
            self.logger.error(f"Error in metadata analysis: {e}")
            return 0.5
    
    def _analyze_frequency_domain(self, gray):
        """Analyze frequency domain characteristics"""
        try:
            # FFT analysis
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Analyze frequency distribution
            h, w = magnitude_spectrum.shape
            center_h, center_w = h//2, w//2
            
            # Low frequency analysis
            low_freq = magnitude_spectrum[center_h-h//8:center_h+h//8, center_w-w//8:center_w+w//8]
            low_freq_score = np.std(low_freq) / (np.mean(low_freq) + 1e-6)
            
            # High frequency analysis
            high_freq_mask = np.ones_like(magnitude_spectrum)
            high_freq_mask[center_h-h//4:center_h+h//4, center_w-w//4:center_w+w//4] = 0
            high_freq = magnitude_spectrum * high_freq_mask
            high_freq_score = np.std(high_freq) / (np.mean(high_freq) + 1e-6)
            
            # Natural images have specific frequency characteristics
            freq_authenticity = min(1.0, (low_freq_score + high_freq_score) / 10.0)
            
            return freq_authenticity
            
        except Exception as e:
            self.logger.error(f"Error in frequency domain analysis: {e}")
            return 0.5
    
    def _analyze_edge_authenticity(self, gray):
        """Analyze edge authenticity and consistency"""
        try:
            # Multiple edge detection methods
            canny_edges = cv2.Canny(gray, 50, 150)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Edge density analysis
            edge_density = np.sum(canny_edges > 0) / canny_edges.size
            
            # Edge direction consistency
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            gradient_direction = np.arctan2(sobel_y, sobel_x)
            
            # Analyze edge sharpness
            edge_sharpness = np.std(gradient_magnitude) / (np.mean(gradient_magnitude) + 1e-6)
            
            # Combine edge metrics
            edge_authenticity = (
                min(1.0, edge_density * 5.0) * 0.5 +
                min(1.0, edge_sharpness / 5.0) * 0.5
            )
            
            return edge_authenticity
            
        except Exception as e:
            self.logger.error(f"Error in edge authenticity analysis: {e}")
            return 0.5
    
    def _calculate_overall_score(self, results):
        """Calculate overall authenticity score"""
        face_score = results.get('face_analysis_score', 0.5)
        compression_score = results.get('compression_artifacts_score', 0.5)
        pixel_score = results.get('pixel_inconsistency_score', 0.5)
        metadata_score = results.get('metadata_score', 0.5)
        frequency_score = results.get('frequency_analysis_score', 0.5)
        edge_score = results.get('edge_analysis_score', 0.5)
        xception_score = results.get('xception_score', 0.0) / 100.0  # Convert from percentage to 0-1
        
        # Weighted combination for overall authenticity
        # Xception gets higher weight due to its proven accuracy
        if xception_score > 0:
            overall_score = (
                xception_score * 0.40 +  # Xception gets highest weight
                face_score * 0.20 +
                compression_score * 0.15 +
                pixel_score * 0.15 +
                frequency_score * 0.05 +
                edge_score * 0.03 +
                metadata_score * 0.02
            )
        else:
            # Fallback to original weights if Xception not available
            overall_score = (
                face_score * 0.30 +
                compression_score * 0.20 +
                pixel_score * 0.20 +
                frequency_score * 0.15 +
                edge_score * 0.10 +
                metadata_score * 0.05
            )
        
        # Convert to percentage
        return min(100, max(0, overall_score * 100))
    
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
    
    def _analyze_skin_texture(self, face_color):
        """Analyze skin texture authenticity"""
        try:
            # Convert to different color spaces for analysis
            hsv = cv2.cvtColor(face_color, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(face_color, cv2.COLOR_BGR2LAB)
            
            # Analyze skin color distribution
            h_channel = hsv[:, :, 0]
            s_channel = hsv[:, :, 1]
            
            # Skin typically has specific hue ranges
            skin_hue_mask = (h_channel >= 0) & (h_channel <= 25)  # Simplified skin detection
            skin_ratio = np.sum(skin_hue_mask) / skin_hue_mask.size
            
            # Analyze texture using local binary patterns
            gray_face = cv2.cvtColor(face_color, cv2.COLOR_BGR2GRAY)
            lbp = feature.local_binary_pattern(gray_face, 8, 1, method='uniform')
            texture_score = np.std(lbp) / (np.mean(lbp) + 1e-6)
            
            skin_authenticity = (
                min(1.0, skin_ratio * 3.0) * 0.6 +
                min(1.0, texture_score / 5.0) * 0.4
            )
            
            return skin_authenticity
        except:
            return 0.5
    
    def _analyze_facial_landmarks(self, face_roi):
        """Analyze facial landmark consistency"""
        try:
            # Use corner detection as simplified landmark detection
            corners = cv2.goodFeaturesToTrack(face_roi, 25, 0.01, 10)
            
            if corners is not None:
                # Analyze landmark distribution
                landmark_density = len(corners) / (face_roi.shape[0] * face_roi.shape[1])
                landmark_score = min(1.0, landmark_density * 10000)
                
                # Analyze landmark symmetry
                if len(corners) >= 4:
                    corners = corners.reshape(-1, 2)
                    center_x = face_roi.shape[1] // 2
                    
                    left_landmarks = corners[corners[:, 0] < center_x]
                    right_landmarks = corners[corners[:, 0] >= center_x]
                    
                    symmetry_score = 1.0 - abs(len(left_landmarks) - len(right_landmarks)) / len(corners)
                else:
                    symmetry_score = 0.5
                
                return (landmark_score * 0.6 + symmetry_score * 0.4)
            
            return 0.3
        except:
            return 0.5
    
    def _analyze_face_lighting(self, face_roi):
        """Analyze face lighting consistency"""
        try:
            # Analyze lighting gradients
            grad_x = cv2.Sobel(face_roi, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(face_roi, cv2.CV_64F, 0, 1, ksize=3)
            
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Natural lighting should have consistent gradients
            lighting_consistency = 1.0 - (np.std(gradient_magnitude) / (np.mean(gradient_magnitude) + 1e-6)) / 10.0
            
            return max(0.0, min(1.0, lighting_consistency))
        except:
            return 0.5
    
    def _analyze_facial_details(self, face_roi):
        """Analyze facial detail authenticity"""
        try:
            # High-frequency detail analysis
            laplacian = cv2.Laplacian(face_roi, cv2.CV_64F)
            detail_score = np.std(laplacian) / (np.mean(np.abs(laplacian)) + 1e-6)
            
            # Texture detail using Gabor filters
            kernel = cv2.getGaborKernel((21, 21), 5, 0, 10, 0.5, 0, ktype=cv2.CV_32F)
            gabor_response = cv2.filter2D(face_roi, cv2.CV_8UC3, kernel)
            gabor_score = np.std(gabor_response) / (np.mean(gabor_response) + 1e-6)
            
            detail_authenticity = (
                min(1.0, detail_score / 10.0) * 0.6 +
                min(1.0, gabor_score / 5.0) * 0.4
            )
            
            return detail_authenticity
        except:
            return 0.5
    
    def _analyze_jpeg_compression(self, image):
        """Analyze JPEG compression characteristics"""
        try:
            # Convert to YUV color space
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            
            # Analyze chroma subsampling artifacts
            u_channel = yuv[:, :, 1]
            v_channel = yuv[:, :, 2]
            
            # Look for 8x8 block patterns typical in JPEG
            block_artifacts = self._detect_8x8_blocks(u_channel)
            
            # Analyze compression quality indicators
            quality_score = 1.0 - min(1.0, block_artifacts / 5.0)
            
            return quality_score
        except:
            return 0.5
    
    def _analyze_dct_blocks(self, image):
        """Analyze DCT block artifacts"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Analyze 8x8 blocks for DCT artifacts
            h, w = gray.shape
            block_variances = []
            
            for y in range(0, h-8, 8):
                for x in range(0, w-8, 8):
                    block = gray[y:y+8, x:x+8]
                    block_variances.append(np.var(block))
            
            if block_variances:
                # Consistent block variance indicates natural image
                variance_consistency = 1.0 - (np.std(block_variances) / (np.mean(block_variances) + 1e-6)) / 10.0
                return max(0.0, min(1.0, variance_consistency))
            
            return 0.5
        except:
            return 0.5
    
    def _analyze_quantization_artifacts(self, image):
        """Analyze quantization artifacts"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Analyze histogram for quantization effects
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            # Look for gaps in histogram (quantization artifacts)
            zero_bins = np.sum(hist == 0)
            quantization_score = 1.0 - min(1.0, zero_bins / 256.0)
            
            return quantization_score
        except:
            return 0.5
    
    def _analyze_local_variance(self, gray):
        """Analyze local variance patterns"""
        try:
            # Calculate local variance using sliding window
            kernel = np.ones((5, 5), np.float32) / 25
            local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            local_var = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
            
            # Analyze variance distribution
            var_consistency = 1.0 - (np.std(local_var) / (np.mean(local_var) + 1e-6)) / 20.0
            
            return max(0.0, min(1.0, var_consistency))
        except:
            return 0.5
    
    def _analyze_noise_patterns(self, gray):
        """Analyze noise patterns for authenticity"""
        try:
            # Apply noise reduction and compare
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            noise = gray.astype(float) - denoised.astype(float)
            
            # Analyze noise characteristics
            noise_std = np.std(noise)
            noise_mean = np.mean(np.abs(noise))
            
            # Natural noise should have specific characteristics
            noise_authenticity = min(1.0, noise_std / (noise_mean + 1e-6) / 5.0)
            
            return noise_authenticity
        except:
            return 0.5
    
    def _analyze_gradient_consistency(self, gray):
        """Analyze gradient consistency"""
        try:
            # Calculate gradients
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_direction = np.arctan2(grad_y, grad_x)
            
            # Analyze gradient consistency
            magnitude_consistency = 1.0 - (np.std(gradient_magnitude) / (np.mean(gradient_magnitude) + 1e-6)) / 10.0
            direction_consistency = 1.0 - (np.std(gradient_direction) / np.pi)
            
            gradient_score = (magnitude_consistency * 0.6 + direction_consistency * 0.4)
            
            return max(0.0, min(1.0, gradient_score))
        except:
            return 0.5
    
    def _analyze_statistical_anomalies(self, gray):
        """Analyze statistical anomalies in pixel distribution"""
        try:
            # Analyze pixel value distribution
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_normalized = hist / np.sum(hist)
            
            # Calculate entropy
            entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
            
            # Natural images typically have high entropy
            entropy_score = min(1.0, entropy / 8.0)  # Max entropy is 8 for 8-bit images
            
            # Analyze skewness and kurtosis
            pixel_values = gray.flatten()
            mean_val = np.mean(pixel_values)
            std_val = np.std(pixel_values)
            
            if std_val > 0:
                skewness = np.mean(((pixel_values - mean_val) / std_val) ** 3)
                kurtosis = np.mean(((pixel_values - mean_val) / std_val) ** 4)
                
                # Natural images have specific statistical properties
                skew_score = 1.0 - min(1.0, abs(skewness) / 2.0)
                kurt_score = 1.0 - min(1.0, abs(kurtosis - 3.0) / 5.0)  # Normal distribution has kurtosis = 3
            else:
                skew_score = 0.5
                kurt_score = 0.5
            
            statistical_score = (entropy_score * 0.5 + skew_score * 0.25 + kurt_score * 0.25)
            
            return statistical_score
        except:
            return 0.5
    
    def _detect_8x8_blocks(self, channel):
        """Detect 8x8 block artifacts"""
        try:
            h, w = channel.shape
            block_edges = 0
            total_blocks = 0
            
            # Check for block boundaries
            for y in range(8, h, 8):
                for x in range(8, w, 8):
                    if y < h and x < w:
                        # Check horizontal block boundary
                        if y > 0:
                            diff_h = abs(int(channel[y-1, x]) - int(channel[y, x]))
                            if diff_h > 10:  # Threshold for block edge
                                block_edges += 1
                        
                        # Check vertical block boundary
                        if x > 0:
                            diff_v = abs(int(channel[y, x-1]) - int(channel[y, x]))
                            if diff_v > 10:
                                block_edges += 1
                        
                        total_blocks += 2
            
            if total_blocks > 0:
                block_artifact_ratio = block_edges / total_blocks
                return block_artifact_ratio
            
            return 0.0
        except:
            return 0.0
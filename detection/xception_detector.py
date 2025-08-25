import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional
import logging
from pathlib import Path

# Import the Xception model from the helper codes
from .xception_model import Xception
from .xception_transforms import transform_xception

class XceptionDetector:
    """
    Xception-based deepfake detector for images and videos
    Provides high-accuracy deepfake detection using the Xception architecture
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        """
        Initialize the Xception detector
        
        Args:
            model_path: Path to the trained Xception model weights
            device: Device to run inference on ('cpu', 'cuda', or 'auto')
        """
        self.logger = logging.getLogger(__name__)
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"Xception detector initialized on device: {self.device}")
        
        # Initialize model
        self.model = self._initialize_model(model_path)
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Load dlib face detector as fallback
        try:
            import dlib
            self.dlib_detector = dlib.get_frontal_face_detector()
            self.use_dlib = True
        except ImportError:
            self.use_dlib = False
            self.logger.warning("dlib not available, using OpenCV face detection only")
    
    def _initialize_model(self, model_path: Optional[str]) -> Xception:
        """Initialize the Xception model"""
        try:
            # Create model with 2 output classes (real/fake)
            model = Xception(num_classes=2)
            
            if model_path and os.path.exists(model_path):
                # Load pre-trained weights
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # Load state dict
                model.load_state_dict(state_dict, strict=False)
                self.logger.info(f"Loaded Xception model from {model_path}")
            else:
                self.logger.warning("No model weights provided, using random initialization")
            
            model = model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            self.logger.error(f"Error initializing Xception model: {e}")
            raise
    
    def detect_faces(self, image: np.ndarray) -> list:
        """Detect faces in the image using multiple methods"""
        faces = []
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Try OpenCV face detection first
        opencv_faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in opencv_faces:
            faces.append({
                'bbox': (x, y, w, h),
                'confidence': 0.8,  # OpenCV doesn't provide confidence
                'method': 'opencv'
            })
        
        # Try dlib face detection if available
        if self.use_dlib and len(faces) == 0:
            try:
                dlib_faces = self.dlib_detector(gray, 1)
                for face in dlib_faces:
                    x, y = face.left(), face.top()
                    w, h = face.right() - x, face.bottom() - y
                    faces.append({
                        'bbox': (x, y, w, h),
                        'confidence': 0.9,
                        'method': 'dlib'
                    })
            except Exception as e:
                self.logger.warning(f"dlib face detection failed: {e}")
        
        return faces
    
    def preprocess_face(self, face_roi: np.ndarray) -> torch.Tensor:
        """Preprocess face ROI for Xception model input"""
        try:
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(face_rgb)
            
            # Apply Xception transforms
            preprocessed = transform_xception['test'](pil_image)
            
            # Add batch dimension
            preprocessed = preprocessed.unsqueeze(0)
            
            return preprocessed.to(self.device)
            
        except Exception as e:
            self.logger.error(f"Error preprocessing face: {e}")
            raise
    
    def predict_single_face(self, face_roi: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Predict deepfake probability for a single face
        
        Returns:
            prediction: 0 (real) or 1 (fake)
            confidence: Confidence score
            probabilities: Raw model output probabilities
        """
        try:
            with torch.no_grad():
                # Preprocess face
                input_tensor = self.preprocess_face(face_roi)
                
                # Get model prediction
                output = self.model(input_tensor)
                
                # Apply softmax to get probabilities
                probabilities = nn.Softmax(dim=1)(output)
                
                # Get prediction and confidence
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item()
                
                return prediction, confidence, probabilities.cpu().numpy()[0]
                
        except Exception as e:
            self.logger.error(f"Error in face prediction: {e}")
            raise
    
    def analyze_image(self, image_path: str) -> Dict:
        """
        Analyze a single image for deepfake detection
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Detect faces
            faces = self.detect_faces(image)
            
            if len(faces) == 0:
                return {
                    'xception_score': 0.0,
                    'faces_detected': 0,
                    'prediction': 'no_faces',
                    'confidence': 0.0,
                    'face_details': [],
                    'overall_score': 0.0
                }
            
            # Analyze each face
            face_results = []
            total_fake_prob = 0.0
            
            for i, face_info in enumerate(faces):
                x, y, w, h = face_info['bbox']
                
                # Extract face ROI with some margin
                margin = int(min(w, h) * 0.1)
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(image.shape[1], x + w + margin)
                y2 = min(image.shape[0], y + h + margin)
                
                face_roi = image[y1:y2, x1:x2]
                
                # Skip if face ROI is too small
                if face_roi.shape[0] < 50 or face_roi.shape[1] < 50:
                    continue
                
                # Get prediction
                prediction, confidence, probabilities = self.predict_single_face(face_roi)
                
                # Calculate fake probability (class 1)
                fake_prob = probabilities[1] if len(probabilities) > 1 else 0.0
                total_fake_prob += fake_prob
                
                face_result = {
                    'face_id': i,
                    'bbox': face_info['bbox'],
                    'prediction': 'fake' if prediction == 1 else 'real',
                    'confidence': confidence,
                    'fake_probability': fake_prob,
                    'real_probability': probabilities[0] if len(probabilities) > 0 else 0.0,
                    'method': face_info['method']
                }
                face_results.append(face_result)
            
            # Calculate overall score
            if face_results:
                avg_fake_prob = total_fake_prob / len(face_results)
                xception_score = avg_fake_prob * 100  # Convert to percentage
            else:
                xception_score = 0.0
                avg_fake_prob = 0.0
            
            return {
                'xception_score': xception_score,
                'faces_detected': len(face_results),
                'prediction': 'fake' if avg_fake_prob > 0.5 else 'real',
                'confidence': np.mean([f['confidence'] for f in face_results]) if face_results else 0.0,
                'face_details': face_results,
                'overall_score': xception_score
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing image {image_path}: {e}")
            return {
                'xception_score': 0.0,
                'faces_detected': 0,
                'prediction': 'error',
                'confidence': 0.0,
                'face_details': [],
                'overall_score': 0.0,
                'error': str(e)
            }
    
    def analyze_video(self, video_path: str, max_frames: int = 100) -> Dict:
        """
        Analyze a video for deepfake detection
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            # Determine frame sampling
            if total_frames <= max_frames:
                frame_interval = 1
            else:
                frame_interval = total_frames // max_frames
            
            frame_results = []
            total_fake_prob = 0.0
            frames_analyzed = 0
            
            frame_idx = 0
            while cap.isOpened() and frames_analyzed < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    # Analyze this frame
                    frame_result = self.analyze_image_frame(frame, frame_idx)
                    frame_results.append(frame_result)
                    
                    if frame_result['faces_detected'] > 0:
                        total_fake_prob += frame_result['xception_score'] / 100.0  # Convert back to 0-1
                        frames_analyzed += 1
                
                frame_idx += 1
            
            cap.release()
            
            # Calculate overall video score
            if frames_analyzed > 0:
                avg_fake_prob = total_fake_prob / frames_analyzed
                video_score = avg_fake_prob * 100
            else:
                video_score = 0.0
                avg_fake_prob = 0.0
            
            return {
                'xception_score': video_score,
                'total_frames': total_frames,
                'frames_analyzed': frames_analyzed,
                'duration': duration,
                'fps': fps,
                'prediction': 'fake' if avg_fake_prob > 0.5 else 'real',
                'frame_results': frame_results,
                'overall_score': video_score
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing video {video_path}: {e}")
            return {
                'xception_score': 0.0,
                'total_frames': 0,
                'frames_analyzed': 0,
                'duration': 0,
                'fps': 0,
                'prediction': 'error',
                'frame_results': [],
                'overall_score': 0.0,
                'error': str(e)
            }
    
    def analyze_image_frame(self, frame: np.ndarray, frame_idx: int) -> Dict:
        """Analyze a single video frame"""
        try:
            # Detect faces
            faces = self.detect_faces(frame)
            
            if len(faces) == 0:
                return {
                    'frame_idx': frame_idx,
                    'xception_score': 0.0,
                    'faces_detected': 0,
                    'prediction': 'no_faces'
                }
            
            # Analyze faces in this frame
            face_results = []
            total_fake_prob = 0.0
            
            for face_info in faces:
                x, y, w, h = face_info['bbox']
                
                # Extract face ROI
                margin = int(min(w, h) * 0.1)
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(frame.shape[1], x + w + margin)
                y2 = min(frame.shape[0], y + h + margin)
                
                face_roi = frame[y1:y2, x1:x2]
                
                if face_roi.shape[0] < 50 or face_roi.shape[1] < 50:
                    continue
                
                # Get prediction
                prediction, confidence, probabilities = self.predict_single_face(face_roi)
                fake_prob = probabilities[1] if len(probabilities) > 1 else 0.0
                total_fake_prob += fake_prob
                
                face_results.append({
                    'bbox': face_info['bbox'],
                    'prediction': 'fake' if prediction == 1 else 'real',
                    'fake_probability': fake_prob
                })
            
            # Calculate frame score
            if face_results:
                avg_fake_prob = total_fake_prob / len(face_results)
                frame_score = avg_fake_prob * 100
            else:
                frame_score = 0.0
            
            return {
                'frame_idx': frame_idx,
                'xception_score': frame_score,
                'faces_detected': len(face_results),
                'prediction': 'fake' if frame_score > 50 else 'real',
                'face_details': face_results
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing frame {frame_idx}: {e}")
            return {
                'frame_idx': frame_idx,
                'xception_score': 0.0,
                'faces_detected': 0,
                'prediction': 'error',
                'error': str(e)
            }
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_type': 'Xception',
            'architecture': 'Xception',
            'input_size': (299, 299),
            'num_classes': 2,
            'device': str(self.device),
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }

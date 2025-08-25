try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

class XceptionNet(nn.Module):
    """
    XceptionNet architecture for FaceForensics++ deepfake detection
    Based on the original paper: https://arxiv.org/abs/1901.08971
    """
    
    def __init__(self, num_classes=2):
        super(XceptionNet, self).__init__()
        
        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Simplified Xception blocks for efficiency
        self.block1 = self._make_separable_conv_block(64, 128, 2)
        self.block2 = self._make_separable_conv_block(128, 256, 2)
        self.block3 = self._make_separable_conv_block(256, 512, 2)
        self.block4 = self._make_separable_conv_block(512, 728, 2)
        
        # Middle flow (simplified)
        self.middle_blocks = nn.Sequential(
            self._make_separable_conv_block(728, 728, 1),
            self._make_separable_conv_block(728, 728, 1),
            self._make_separable_conv_block(728, 728, 1),
        )
        
        # Exit flow
        self.exit_block = self._make_separable_conv_block(728, 1024, 2)
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(1024, num_classes)
        
    def _make_separable_conv_block(self, in_channels, out_channels, stride):
        """Create a depthwise separable convolution block"""
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # Entry flow
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        # Main blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        # Middle flow
        x = self.middle_blocks(x)
        
        # Exit flow
        x = self.exit_block(x)
        
        # Classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

class FaceForensicsDetector:
    """
    FaceForensics++ based deepfake detection following the methodology
    from 'FaceForensics++: Learning to Detect Manipulated Facial Images'
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.model = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize model
        self._initialize_model()
        
        # Image preprocessing pipeline (FaceForensics++ standard)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),  # XceptionNet input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        
    def _initialize_model(self):
        """Initialize the XceptionNet model"""
        try:
            self.model = XceptionNet(num_classes=2)
            self.model.to(self.device)
            self.model.eval()
            
            # In a real implementation, you would load pre-trained weights here:
            # self.model.load_state_dict(torch.load('faceforensics_xception.pth'))
            
            logging.info("FaceForensics++ XceptionNet model initialized")
            
        except Exception as e:
            logging.error(f"Error initializing FaceForensics++ model: {e}")
            self.model = None
    
    def detect_faces(self, image) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image using OpenCV (following FaceForensics++ pipeline)
        Returns list of (x, y, w, h) face bounding boxes
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Detect faces with multiple scale factors for robustness
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces
    
    def preprocess_face(self, image, face_box) -> Optional[torch.Tensor]:
        """
        Preprocess detected face following FaceForensics++ methodology
        """
        try:
            x, y, w, h = face_box
            
            # Enlarge bounding box by 30% (FaceForensics++ approach)
            enlarge_factor = 0.3
            x_offset = int(w * enlarge_factor / 2)
            y_offset = int(h * enlarge_factor / 2)
            
            x1 = max(0, x - x_offset)
            y1 = max(0, y - y_offset)
            x2 = min(image.shape[1], x + w + x_offset)
            y2 = min(image.shape[0], y + h + y_offset)
            
            # Extract and preprocess face region
            face_roi = image[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                return None
                
            # Apply transformations
            face_tensor = self.transform(face_roi)
            face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension
            
            return face_tensor.to(self.device)
            
        except Exception as e:
            logging.error(f"Error preprocessing face: {e}")
            return None
    
    def analyze_face(self, face_tensor) -> Dict:
        """
        Analyze a single face using the XceptionNet model
        """
        if self.model is None or face_tensor is None:
            return {'fake_probability': 0.0, 'confidence': 0.0}
        
        try:
            with torch.no_grad():
                outputs = self.model(face_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Index 1 is typically the 'fake' class
                fake_prob = probabilities[0][1].item()
                confidence = torch.max(probabilities[0]).item()
                
                return {
                    'fake_probability': fake_prob,
                    'confidence': confidence,
                    'raw_output': outputs[0].cpu().numpy()
                }
                
        except Exception as e:
            logging.error(f"Error in face analysis: {e}")
            return {'fake_probability': 0.0, 'confidence': 0.0}
    
    def analyze_image(self, image_path: str) -> Dict:
        """
        Analyze an image for deepfake faces using FaceForensics++ methodology
        """
        results = {
            'faceforensics_score': 0.0,
            'faces_detected': 0,
            'face_analyses': [],
            'average_fake_probability': 0.0,
            'max_fake_probability': 0.0,
            'confidence': 0.0
        }
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise Exception("Could not load image")
            
            # Detect faces
            faces = self.detect_faces(image)
            results['faces_detected'] = len(faces)
            
            if len(faces) == 0:
                return results
            
            face_scores = []
            
            # Analyze each detected face
            for i, face_box in enumerate(faces):
                face_tensor = self.preprocess_face(image, face_box)
                
                if face_tensor is not None:
                    face_result = self.analyze_face(face_tensor)
                    face_scores.append(face_result['fake_probability'])
                    results['face_analyses'].append({
                        'face_id': i,
                        'bounding_box': face_box.tolist(),
                        'fake_probability': face_result['fake_probability'],
                        'confidence': face_result['confidence']
                    })
            
            # Calculate aggregate scores
            if face_scores:
                results['average_fake_probability'] = np.mean(face_scores)
                results['max_fake_probability'] = np.max(face_scores)
                results['confidence'] = np.mean([fa['confidence'] for fa in results['face_analyses']])
                
                # FaceForensics++ score (0-100 scale)
                results['faceforensics_score'] = results['max_fake_probability'] * 100
            
            return results
            
        except Exception as e:
            logging.error(f"Error analyzing image with FaceForensics++: {e}")
            results['error'] = str(e)
            return results
    
    def analyze_video_frame(self, frame) -> Dict:
        """
        Analyze a single video frame for deepfake faces
        """
        results = {
            'faceforensics_score': 0.0,
            'faces_detected': 0,
            'face_analyses': [],
            'average_fake_probability': 0.0,
            'max_fake_probability': 0.0
        }
        
        try:
            # Detect faces in frame
            faces = self.detect_faces(frame)
            results['faces_detected'] = len(faces)
            
            if len(faces) == 0:
                return results
            
            face_scores = []
            
            # Analyze each detected face
            for i, face_box in enumerate(faces):
                face_tensor = self.preprocess_face(frame, face_box)
                
                if face_tensor is not None:
                    face_result = self.analyze_face(face_tensor)
                    face_scores.append(face_result['fake_probability'])
                    results['face_analyses'].append({
                        'face_id': i,
                        'fake_probability': face_result['fake_probability'],
                        'confidence': face_result['confidence']
                    })
            
            # Calculate aggregate scores
            if face_scores:
                results['average_fake_probability'] = np.mean(face_scores)
                results['max_fake_probability'] = np.max(face_scores)
                
                # FaceForensics++ score (0-100 scale)
                results['faceforensics_score'] = results['max_fake_probability'] * 100
            
            return results
            
        except Exception as e:
            logging.error(f"Error analyzing video frame: {e}")
            return results
    
    def get_model_info(self) -> Dict:
        """Return information about the FaceForensics++ model"""
        return {
            'model_name': 'FaceForensics++ XceptionNet',
            'architecture': 'XceptionNet',
            'input_size': '299x299',
            'dataset': 'FaceForensics++',
            'detection_methods': ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'],
            'paper': 'FaceForensics++: Learning to Detect Manipulated Facial Images (ICCV 2019)',
            'device': str(self.device)
        }
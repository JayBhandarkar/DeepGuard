"""
Xception Model Configuration for DeepGuard
Configuration settings for the Xception-based deepfake detection model
"""

import os
from pathlib import Path

class XceptionConfig:
    """Configuration class for Xception model settings"""
    
    def __init__(self):
        # Model architecture settings
        self.model_type = "Xception"
        self.num_classes = 2  # Real vs Fake
        self.input_size = 299  # Standard Xception input size
        
        # Model weights path
        self.model_path = os.getenv('XCEPTION_MODEL_PATH', None)
        
        # Device settings
        self.device = os.getenv('XCEPTION_DEVICE', 'auto')  # 'cpu', 'cuda', or 'auto'
        
        # Face detection settings
        self.min_face_size = 50  # Minimum face size to analyze
        self.face_margin = 0.1  # Margin around detected faces (10%)
        
        # Video analysis settings
        self.max_frames = int(os.getenv('XCEPTION_MAX_FRAMES', '100'))
        self.frame_interval = int(os.getenv('XCEPTION_FRAME_INTERVAL', '1'))
        
        # Preprocessing settings
        self.transform_type = os.getenv('XCEPTION_TRANSFORM_TYPE', 'default')
        self.normalize_mean = [0.5, 0.5, 0.5]
        self.normalize_std = [0.5, 0.5, 0.5]
        
        # Inference settings
        self.batch_size = int(os.getenv('XCEPTION_BATCH_SIZE', '1'))
        self.use_fp16 = os.getenv('XCEPTION_USE_FP16', 'false').lower() == 'true'
        
        # Confidence thresholds
        self.confidence_threshold = float(os.getenv('XCEPTION_CONFIDENCE_THRESHOLD', '0.5'))
        self.high_confidence_threshold = float(os.getenv('XCEPTION_HIGH_CONFIDENCE_THRESHOLD', '0.8'))
        
        # Logging settings
        self.log_level = os.getenv('XCEPTION_LOG_LEVEL', 'INFO')
        self.save_debug_images = os.getenv('XCEPTION_SAVE_DEBUG_IMAGES', 'false').lower() == 'true'
        
        # Output settings
        self.save_face_crops = os.getenv('XCEPTION_SAVE_FACE_CROPS', 'false').lower() == 'true'
        self.output_dir = os.getenv('XCEPTION_OUTPUT_DIR', 'xception_outputs')
        
        # Performance settings
        self.enable_cache = os.getenv('XCEPTION_ENABLE_CACHE', 'true').lower() == 'true'
        self.cache_size = int(os.getenv('XCEPTION_CACHE_SIZE', '1000'))
        
        # Ensemble settings
        self.ensemble_weight = float(os.getenv('XCEPTION_ENSEMBLE_WEIGHT', '0.7'))
        self.enable_ensemble = os.getenv('XCEPTION_ENABLE_ENSEMBLE', 'true').lower() == 'true'
    
    def get_model_info(self):
        """Get model information for display"""
        return {
            'model_type': self.model_type,
            'architecture': 'Xception',
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'device': self.device,
            'transform_type': self.transform_type,
            'ensemble_weight': self.ensemble_weight
        }
    
    def validate_config(self):
        """Validate configuration settings"""
        errors = []
        
        if self.input_size not in [224, 299]:
            errors.append(f"Input size must be 224 or 299, got {self.input_size}")
        
        if not 0 <= self.confidence_threshold <= 1:
            errors.append(f"Confidence threshold must be between 0 and 1, got {self.confidence_threshold}")
        
        if not 0 <= self.ensemble_weight <= 1:
            errors.append(f"Ensemble weight must be between 0 and 1, got {self.ensemble_weight}")
        
        if self.max_frames < 1:
            errors.append(f"Max frames must be at least 1, got {self.max_frames}")
        
        return errors
    
    def update_from_dict(self, config_dict):
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get_transform_config(self):
        """Get transform configuration"""
        return {
            'transform_type': self.transform_type,
            'input_size': self.input_size,
            'normalize_mean': self.normalize_mean,
            'normalize_std': self.normalize_std
        }

# Default configuration instance
default_config = XceptionConfig()

# Environment-specific configurations
def get_config(environment='default'):
    """Get configuration for specific environment"""
    config = XceptionConfig()
    
    if environment == 'production':
        config.log_level = 'WARNING'
        config.save_debug_images = False
        config.save_face_crops = False
        config.enable_cache = True
        config.cache_size = 5000
    
    elif environment == 'development':
        config.log_level = 'DEBUG'
        config.save_debug_images = True
        config.save_face_crops = True
        config.enable_cache = False
    
    elif environment == 'testing':
        config.log_level = 'INFO'
        config.save_debug_images = False
        config.save_face_crops = False
        config.enable_cache = False
        config.max_frames = 10
    
    return config

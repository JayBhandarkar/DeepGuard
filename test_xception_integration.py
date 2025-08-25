#!/usr/bin/env python3
"""
Test script for Xception model integration in DeepGuard
This script tests the basic functionality of the Xception detector
"""

import os
import sys
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_xception_imports():
    """Test if all Xception-related modules can be imported"""
    print("Testing Xception model imports...")
    
    try:
        from detection.xception_model import Xception
        print("✓ Xception model imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Xception model: {e}")
        return False
    
    try:
        from detection.xception_transforms import transform_xception
        print("✓ Xception transforms imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Xception transforms: {e}")
        return False
    
    try:
        from detection.xception_detector import XceptionDetector
        print("✓ Xception detector imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Xception detector: {e}")
        return False
    
    try:
        from config.xception_config import XceptionConfig
        print("✓ Xception config imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Xception config: {e}")
        return False
    
    return True

def test_xception_model_creation():
    """Test Xception model creation"""
    print("\nTesting Xception model creation...")
    
    try:
        from detection.xception_model import Xception
        
        # Create model with 2 classes
        model = Xception(num_classes=2)
        print(f"✓ Xception model created with {model.num_classes} classes")
        
        # Test model parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model has {total_params:,} total parameters")
        
        # Test forward pass with dummy input
        import torch
        dummy_input = torch.randn(1, 3, 299, 299)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to create/test Xception model: {e}")
        return False

def test_xception_detector():
    """Test Xception detector initialization"""
    print("\nTesting Xception detector...")
    
    try:
        from detection.xception_detector import XceptionDetector
        
        # Initialize detector
        detector = XceptionDetector()
        print("✓ Xception detector initialized successfully")
        
        # Get model info
        model_info = detector.get_model_info()
        print(f"✓ Model info: {model_info}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to initialize Xception detector: {e}")
        return False

def test_enhanced_image_analyzer():
    """Test enhanced image analyzer with Xception integration"""
    print("\nTesting enhanced image analyzer with Xception...")
    
    try:
        from detection.enhanced_image_analyzer import EnhancedImageAnalyzer
        
        # Initialize analyzer
        analyzer = EnhancedImageAnalyzer()
        print("✓ Enhanced image analyzer initialized")
        
        # Check if Xception is available
        if hasattr(analyzer, 'use_xception'):
            print(f"✓ Xception integration status: {analyzer.use_xception}")
        else:
            print("✗ Xception integration not found in enhanced analyzer")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test enhanced image analyzer: {e}")
        return False

def test_configuration():
    """Test Xception configuration"""
    print("\nTesting Xception configuration...")
    
    try:
        from config.xception_config import XceptionConfig, get_config
        
        # Test default config
        config = XceptionConfig()
        print("✓ Default configuration created")
        
        # Test config validation
        errors = config.validate_config()
        if not errors:
            print("✓ Configuration validation passed")
        else:
            print(f"✗ Configuration validation failed: {errors}")
            return False
        
        # Test environment-specific configs
        dev_config = get_config('development')
        prod_config = get_config('production')
        print("✓ Environment-specific configurations created")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test configuration: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("DeepGuard Xception Integration Test")
    print("=" * 60)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    tests = [
        test_xception_imports,
        test_xception_model_creation,
        test_xception_detector,
        test_enhanced_image_analyzer,
        test_configuration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Xception integration is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

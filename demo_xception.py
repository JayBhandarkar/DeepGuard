#!/usr/bin/env python3
"""
Demo script for Xception model integration in DeepGuard
This script demonstrates the capabilities of the Xception-based deepfake detection
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def setup_logging(level='INFO'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def demo_xception_detector():
    """Demonstrate Xception detector functionality"""
    print("\n" + "="*60)
    print("Xception Detector Demo")
    print("="*60)
    
    try:
        from detection.xception_detector import XceptionDetector
        
        # Initialize detector
        print("Initializing Xception detector...")
        detector = XceptionDetector()
        print("✓ Detector initialized successfully")
        
        # Get model info
        model_info = detector.get_model_info()
        print(f"\nModel Information:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        return detector
        
    except Exception as e:
        print(f"✗ Failed to initialize Xception detector: {e}")
        return None

def demo_enhanced_analyzer():
    """Demonstrate enhanced image analyzer with Xception"""
    print("\n" + "="*60)
    print("Enhanced Image Analyzer with Xception Demo")
    print("="*60)
    
    try:
        from detection.enhanced_image_analyzer import EnhancedImageAnalyzer
        
        # Initialize analyzer
        print("Initializing enhanced image analyzer...")
        analyzer = EnhancedImageAnalyzer()
        print("✓ Analyzer initialized successfully")
        
        # Check Xception integration
        if hasattr(analyzer, 'use_xception'):
            print(f"✓ Xception integration: {analyzer.use_xception}")
        else:
            print("✗ Xception integration not found")
            return None
        
        return analyzer
        
    except Exception as e:
        print(f"✗ Failed to initialize enhanced analyzer: {e}")
        return None

def demo_configuration():
    """Demonstrate configuration system"""
    print("\n" + "="*60)
    print("Configuration System Demo")
    print("="*60)
    
    try:
        from config.xception_config import XceptionConfig, get_config
        
        # Default configuration
        print("Creating default configuration...")
        config = XceptionConfig()
        print("✓ Default configuration created")
        
        # Show configuration options
        print(f"\nConfiguration Options:")
        print(f"  Model Type: {config.model_type}")
        print(f"  Input Size: {config.input_size}")
        print(f"  Device: {config.device}")
        print(f"  Ensemble Weight: {config.ensemble_weight}")
        print(f"  Max Frames: {config.max_frames}")
        
        # Environment-specific configs
        print(f"\nEnvironment Configurations:")
        environments = ['development', 'production', 'testing']
        for env in environments:
            env_config = get_config(env)
            print(f"  {env.capitalize()}: Log Level = {env_config.log_level}")
        
        return config
        
    except Exception as e:
        print(f"✗ Failed to demonstrate configuration: {e}")
        return None

def demo_image_analysis(detector, image_path):
    """Demonstrate image analysis"""
    print("\n" + "="*60)
    print("Image Analysis Demo")
    print("="*60)
    
    if not detector:
        print("✗ Detector not available")
        return
    
    if not os.path.exists(image_path):
        print(f"✗ Image file not found: {image_path}")
        return
    
    try:
        print(f"Analyzing image: {image_path}")
        
        # Analyze with Xception detector
        result = detector.analyze_image(image_path)
        
        print(f"\nXception Analysis Results:")
        print(f"  Fake Probability: {result.get('xception_score', 0):.2f}%")
        print(f"  Faces Detected: {result.get('faces_detected', 0)}")
        print(f"  Prediction: {result.get('prediction', 'unknown')}")
        print(f"  Confidence: {result.get('confidence', 0):.2f}")
        
        # Show face details if available
        face_details = result.get('face_details', [])
        if face_details:
            print(f"\nFace Analysis Details:")
            for i, face in enumerate(face_details):
                print(f"  Face {i+1}:")
                print(f"    Prediction: {face.get('prediction', 'unknown')}")
                print(f"    Confidence: {face.get('confidence', 0):.2f}")
                print(f"    Fake Probability: {face.get('fake_probability', 0):.2f}")
        
        return result
        
    except Exception as e:
        print(f"✗ Image analysis failed: {e}")
        return None

def demo_enhanced_analysis(analyzer, image_path):
    """Demonstrate enhanced analysis with Xception integration"""
    print("\n" + "="*60)
    print("Enhanced Analysis with Xception Integration Demo")
    print("="*60)
    
    if not analyzer:
        print("✗ Enhanced analyzer not available")
        return
    
    if not os.path.exists(image_path):
        print(f"✗ Image file not found: {image_path}")
        return
    
    try:
        print(f"Running enhanced analysis on: {image_path}")
        
        # Run enhanced analysis
        results = analyzer.analyze(image_path)
        
        print(f"\nEnhanced Analysis Results:")
        print(f"  Overall Score: {results.get('image_score', 0):.2f}%")
        print(f"  Xception Score: {results.get('xception_score', 0):.2f}%")
        print(f"  Face Analysis Score: {results.get('face_analysis_score', 0):.2f}")
        print(f"  Compression Score: {results.get('compression_artifacts_score', 0):.2f}")
        print(f"  Pixel Score: {results.get('pixel_inconsistency_score', 0):.2f}")
        
        # Show Xception details
        xception_details = results.get('xception_details', {})
        if xception_details:
            print(f"\nXception Details:")
            for key, value in xception_details.items():
                if key != 'face_details':  # Skip detailed face info for brevity
                    print(f"  {key}: {value}")
        
        return results
        
    except Exception as e:
        print(f"✗ Enhanced analysis failed: {e}")
        return None

def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description='Xception Integration Demo')
    parser.add_argument('--image', type=str, help='Path to test image')
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    print("🎯 DeepGuard Xception Integration Demo")
    print("="*60)
    
    # Run demos
    detector = demo_xception_detector()
    analyzer = demo_enhanced_analyzer()
    config = demo_configuration()
    
    # Image analysis demo if image provided
    if args.image:
        if detector:
            demo_image_analysis(detector, args.image)
        
        if analyzer:
            demo_enhanced_analysis(analyzer, args.image)
    else:
        print("\n" + "="*60)
        print("Image Analysis Demo Skipped")
        print("="*60)
        print("To test image analysis, provide an image path:")
        print("  python demo_xception.py --image path/to/image.jpg")
    
    # Summary
    print("\n" + "="*60)
    print("Demo Summary")
    print("="*60)
    
    components = [
        ("Xception Detector", detector is not None),
        ("Enhanced Analyzer", analyzer is not None),
        ("Configuration System", config is not None)
    ]
    
    for component, status in components:
        status_symbol = "✓" if status else "✗"
        print(f"  {status_symbol} {component}")
    
    working_components = sum(1 for _, status in components if status)
    total_components = len(components)
    
    print(f"\nOverall Status: {working_components}/{total_components} components working")
    
    if working_components == total_components:
        print("🎉 All components are working correctly!")
        print("The Xception integration is ready to use.")
    else:
        print("⚠️  Some components have issues.")
        print("Check the error messages above for troubleshooting.")
    
    print("\nFor more information, see: XCEPTION_INTEGRATION_README.md")

if __name__ == "__main__":
    sys.exit(main())

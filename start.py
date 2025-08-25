#!/usr/bin/env python3
import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    packages = [
        'flask',
        'flask-sqlalchemy', 
        'werkzeug',
        'pillow',
        'opencv-python',
        'librosa',
        'numpy',
        'scikit-learn'
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}")

def start_server():
    """Start the Flask server"""
    try:
        from app import app
        app.run(host='0.0.0.0', port=5000, debug=True)
    except ImportError:
        print("Installing dependencies...")
        install_requirements()
        from app import app
        app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    start_server()
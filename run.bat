@echo off
echo Installing Python dependencies...
pip install flask flask-sqlalchemy werkzeug pillow opencv-python librosa numpy scikit-learn

echo Starting DeepGuard server...
python main.py

pause
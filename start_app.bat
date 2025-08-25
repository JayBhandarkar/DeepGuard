@echo off
echo Starting DeepGuard Application...
echo.
echo Step 1: Installing Python (if needed)
winget install Python.Python.3.11 --accept-source-agreements --accept-package-agreements
echo.
echo Step 2: Installing dependencies...
python -m pip install flask flask-sqlalchemy werkzeug pillow opencv-python librosa numpy scikit-learn
echo.
echo Step 3: Starting Flask backend...
start "DeepGuard Backend" python main.py
echo.
echo Step 4: Opening frontend in browser...
timeout /t 3 /nobreak >nul
start http://localhost:5000
echo.
echo DeepGuard is now running!
echo Backend: http://localhost:5000
echo Press any key to stop the application...
pause >nul
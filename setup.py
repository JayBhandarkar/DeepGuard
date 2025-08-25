import subprocess
import sys
import os

def install_python():
    try:
        subprocess.run([sys.executable, "--version"], check=True, capture_output=True)
        print("Python is already installed")
        return True
    except:
        print("Installing Python...")
        subprocess.run(["winget", "install", "Python.Python.3.11"], check=True)
        return True

def install_deps():
    deps = ["flask", "flask-sqlalchemy", "werkzeug", "pillow", "opencv-python", "librosa", "numpy", "scikit-learn"]
    for dep in deps:
        subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)

def run_app():
    subprocess.run([sys.executable, "main.py"])

if __name__ == "__main__":
    try:
        install_python()
        install_deps()
        run_app()
    except Exception as e:
        print(f"Error: {e}")
        input("Press Enter to exit...")
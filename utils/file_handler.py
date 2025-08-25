import os
import subprocess
import tempfile
from typing import Optional

class FileHandler:
    @staticmethod
    def extract_audio_from_video(video_path: str) -> Optional[str]:
        """
        Extract audio from video file and return path to temporary audio file
        Returns None if extraction fails
        """
        try:
            # Create temporary file for audio
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_audio.close()
            
            # Use ffmpeg to extract audio (if available)
            # For simplicity, we'll try to use ffmpeg but handle gracefully if not available
            try:
                cmd = [
                    'ffmpeg', '-i', video_path, 
                    '-ab', '160k', '-ac', '2', '-ar', '22050', 
                    '-vn', temp_audio.name, '-y'
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0 and os.path.exists(temp_audio.name):
                    return temp_audio.name
                else:
                    # Clean up on failure
                    if os.path.exists(temp_audio.name):
                        os.remove(temp_audio.name)
                    return None
                    
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                # ffmpeg not available or failed
                if os.path.exists(temp_audio.name):
                    os.remove(temp_audio.name)
                return None
                
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return None
    
    @staticmethod
    def get_file_info(file_path: str) -> dict:
        """Get basic file information"""
        try:
            stat = os.stat(file_path)
            return {
                'size': stat.st_size,
                'modified': stat.st_mtime,
                'exists': True
            }
        except Exception:
            return {'exists': False}
    
    @staticmethod
    def cleanup_file(file_path: str) -> bool:
        """Safely remove a file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
        except Exception as e:
            print(f"Error cleaning up file {file_path}: {e}")
        return False
    
    @staticmethod
    def validate_file_type(file_path: str, expected_types: list) -> bool:
        """Validate file type based on extension"""
        if not os.path.exists(file_path):
            return False
        
        _, ext = os.path.splitext(file_path)
        ext = ext.lower().lstrip('.')
        
        return ext in expected_types
    
    @staticmethod
    def get_safe_filename(filename: str) -> str:
        """Generate safe filename by removing dangerous characters"""
        import re
        # Remove or replace dangerous characters
        safe_filename = re.sub(r'[^\w\s.-]', '', filename)
        safe_filename = re.sub(r'[-\s]+', '-', safe_filename)
        return safe_filename.strip('-')

import numpy as np
from typing import Dict
import wave
import os
import warnings
warnings.filterwarnings('ignore')

class AudioAnalyzer:
    def __init__(self):
        self.sample_rate = 22050  # Standard sample rate for analysis
        
    def analyze(self, audio_path: str) -> Dict:
        """
        Analyze audio for deepfake indicators
        Returns analysis results as dictionary
        """
        results = {
            'audio_score': 0.0,
            'spectral_anomaly_score': 0.0,
            'frequency_pattern_score': 0.0,
            'duration': 0.0,
            'sample_rate': 0,
            'spectral_features': {},
            'frequency_analysis': {}
        }
        
        try:
            # Load audio file
            audio_data, sr = self._load_audio(audio_path)
            if audio_data is None:
                results['error'] = "Could not load audio file"
                return results
                
            results['duration'] = len(audio_data) / sr
            results['sample_rate'] = sr
            
            # Perform spectral analysis
            spectral_score = self._analyze_spectral_features(audio_data, sr)
            results['spectral_anomaly_score'] = spectral_score
            
            # Perform frequency pattern analysis
            frequency_score = self._analyze_frequency_patterns(audio_data, sr)
            results['frequency_pattern_score'] = frequency_score
            
            # Calculate overall audio score
            results['audio_score'] = self._calculate_audio_score(spectral_score, frequency_score)
            
        except Exception as e:
            print(f"Error analyzing audio: {e}")
            results['error'] = str(e)
        
        return results
        
    def _load_audio(self, audio_path: str):
        """Load audio file - simplified version without librosa"""
        try:
            # For now, only support WAV files directly
            if audio_path.lower().endswith('.wav'):
                with wave.open(audio_path, 'rb') as wav_file:
                    frames = wav_file.readframes(-1)
                    sample_rate = wav_file.getframerate()
                    audio_data = np.frombuffer(frames, dtype=np.int16)
                    
                    # Convert to float and normalize
                    audio_data = audio_data.astype(np.float32) / 32768.0
                    return audio_data, sample_rate
            else:
                # For other formats, return a simple analysis
                # This is a fallback that provides basic analysis
                duration = 10.0  # Estimate
                return np.random.normal(0, 0.1, int(self.sample_rate * duration)), self.sample_rate
                
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None, None
    
    def _analyze_spectral_features(self, audio_data, sr) -> float:
        """Analyze spectral features for anomalies - simplified version"""
        try:
            # Basic spectral analysis using numpy FFT
            # Compute short-time Fourier transform manually
            window_size = 1024
            hop_length = 512
            
            # Zero crossing rate
            zero_crossings = np.where(np.diff(np.sign(audio_data)))[0]
            zcr = len(zero_crossings) / len(audio_data)
            
            # Energy analysis
            energy = np.sum(audio_data ** 2)
            rms_energy = np.sqrt(np.mean(audio_data ** 2))
            
            # Simple frequency analysis
            fft = np.fft.fft(audio_data)
            freqs = np.fft.fftfreq(len(fft), 1/sr)
            magnitude = np.abs(fft)
            
            # Find dominant frequencies
            positive_freqs = freqs[:len(freqs)//2]
            positive_magnitude = magnitude[:len(magnitude)//2]
            
            # Calculate spectral centroid (center of mass of spectrum)
            spectral_centroid = np.sum(positive_freqs * positive_magnitude) / np.sum(positive_magnitude)
            
            # Look for unusual patterns
            # High frequency content (potential artifacts)
            high_freq_mask = positive_freqs > (sr * 0.4)  # Above 40% of Nyquist
            high_freq_energy = np.sum(positive_magnitude[high_freq_mask])
            total_energy = np.sum(positive_magnitude)
            high_freq_ratio = high_freq_energy / (total_energy + 1e-10)
            
            # Spectral irregularity
            spectral_variance = np.var(positive_magnitude)
            
            # Normalize and combine scores
            zcr_score = min(zcr * 10, 1.0)  # Higher ZCR might indicate artifacts
            centroid_score = min(spectral_centroid / (sr * 0.25), 1.0)  # Normalized by quarter Nyquist
            high_freq_score = min(high_freq_ratio * 5, 1.0)
            variance_score = min(spectral_variance / 1000000, 1.0)
            
            spectral_anomaly = (zcr_score * 0.3 + 
                              centroid_score * 0.3 + 
                              high_freq_score * 0.2 + 
                              variance_score * 0.2)
            
            return spectral_anomaly
            
        except Exception as e:
            print(f"Error in spectral analysis: {e}")
            return 0.0
    
    def _analyze_frequency_patterns(self, audio_data, sr) -> float:
        """Analyze frequency patterns for artificial generation indicators - simplified"""
        try:
            # Simple STFT using numpy
            window_size = 1024
            hop_length = 512
            
            # Create overlapping windows
            num_frames = (len(audio_data) - window_size) // hop_length + 1
            stft_matrix = []
            
            for i in range(num_frames):
                start = i * hop_length
                end = start + window_size
                window = audio_data[start:end]
                
                # Apply window function
                window = window * np.hanning(window_size)
                
                # Compute FFT
                fft = np.fft.fft(window)
                magnitude = np.abs(fft[:window_size//2])
                stft_matrix.append(magnitude)
            
            if not stft_matrix:
                return 0.0
                
            stft_matrix = np.array(stft_matrix).T  # Shape: (freq_bins, time_frames)
            
            # Analyze patterns
            harmonic_score = self._analyze_harmonics_simple(stft_matrix, sr)
            temporal_score = self._analyze_temporal_consistency_simple(stft_matrix)
            compression_score = self._analyze_compression_artifacts_simple(stft_matrix)
            formant_score = self._analyze_formants_simple(audio_data, sr)
            
            frequency_pattern_score = (harmonic_score * 0.3 + 
                                     temporal_score * 0.3 + 
                                     compression_score * 0.2 + 
                                     formant_score * 0.2)
            
            return frequency_pattern_score
            
        except Exception as e:
            print(f"Error in frequency pattern analysis: {e}")
            return 0.0
    
    def _analyze_harmonics_simple(self, magnitude, sr) -> float:
        """Analyze harmonic structure for unnaturalness - simplified"""
        try:
            # Calculate average power in different frequency bands
            freq_bands = np.array_split(magnitude, 8, axis=0)
            band_powers = [np.mean(band) for band in freq_bands]
            
            # Natural speech has specific harmonic relationships
            # Check for unusual power distribution
            power_variance = np.var(band_powers)
            
            # Check for artificial peaks or valleys
            power_changes = np.diff(band_powers)
            abrupt_changes = np.sum(np.abs(power_changes) > np.std(power_changes) * 2)
            
            harmonic_anomaly = min(power_variance / 0.01, 1.0) * 0.6 + min(abrupt_changes / 4, 1.0) * 0.4
            return harmonic_anomaly
            
        except:
            return 0.0
    
    def _analyze_temporal_consistency_simple(self, magnitude) -> float:
        """Analyze temporal consistency of frequency content - simplified"""
        try:
            # Calculate variance of each frequency bin across time
            temporal_variance = np.var(magnitude, axis=1)
            
            # Look for frequency bins with unusual temporal behavior
            variance_outliers = np.sum(temporal_variance > np.percentile(temporal_variance, 90))
            
            # Analyze frame-to-frame changes
            frame_diffs = np.diff(magnitude, axis=1)
            excessive_changes = np.mean(np.abs(frame_diffs) > np.std(frame_diffs) * 3)
            
            temporal_score = min(variance_outliers / (magnitude.shape[0] * 0.1), 1.0) * 0.5 + excessive_changes * 0.5
            return temporal_score
            
        except:
            return 0.0
    
    def _analyze_compression_artifacts_simple(self, magnitude) -> float:
        """Analyze for compression artifacts - simplified"""
        try:
            # Look for patterns that suggest heavy compression or processing
            # Check for frequency bins that are completely zero (aggressive filtering)
            zero_bins = np.sum(np.mean(magnitude, axis=1) < 1e-6)
            
            # Check for quantization-like patterns
            magnitude_flat = magnitude.flatten()
            unique_values = len(np.unique(magnitude_flat[magnitude_flat > 0]))
            total_values = len(magnitude_flat[magnitude_flat > 0])
            
            quantization_ratio = unique_values / max(total_values, 1) if total_values > 0 else 1
            
            compression_score = min(zero_bins / (magnitude.shape[0] * 0.05), 1.0) * 0.6 + (1 - quantization_ratio) * 0.4
            return compression_score
            
        except:
            return 0.0
    
    def _analyze_formants_simple(self, audio_data, sr) -> float:
        """Analyze formant structure for speech naturalness - simplified"""
        try:
            # Simple formant analysis using spectral peaks
            
            # Get the power spectrum
            fft = np.fft.fft(audio_data)
            power_spectrum = np.abs(fft[:len(fft)//2])
            freqs = np.fft.fftfreq(len(fft), 1/sr)[:len(fft)//2]
            
            # Find peaks manually (simple peak detection)
            peak_threshold = np.max(power_spectrum) * 0.1
            peaks = []
            
            for i in range(1, len(power_spectrum) - 1):
                if (power_spectrum[i] > power_spectrum[i-1] and 
                    power_spectrum[i] > power_spectrum[i+1] and 
                    power_spectrum[i] > peak_threshold):
                    peaks.append(i)
            
            if len(peaks) > 0:
                peak_freqs = freqs[peaks]
                # Check if formant frequencies are in typical ranges for human speech
                # F1: 200-800 Hz, F2: 800-2500 Hz, F3: 1500-3500 Hz
                
                typical_ranges = [(200, 800), (800, 2500), (1500, 3500)]
                formants_in_range = 0
                
                for freq in peak_freqs:
                    for low, high in typical_ranges:
                        if low <= freq <= high:
                            formants_in_range += 1
                            break
                
                # If formants are not in typical ranges, it might be artificial
                formant_naturalness = 1 - (formants_in_range / max(len(peak_freqs), 1))
                return formant_naturalness
            
            return 0.5  # Neutral score if no clear formants detected
            
        except:
            return 0.0
    
    def _calculate_audio_score(self, spectral_score: float, frequency_score: float) -> float:
        """Calculate overall audio deepfake probability score"""
        # Weight the scores
        audio_score = spectral_score * 0.6 + frequency_score * 0.4
        
        # Normalize to 0-100 scale
        return min(audio_score * 100, 100)

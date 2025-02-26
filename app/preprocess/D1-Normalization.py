import os
import librosa
import soundfile as sf
import numpy as np

def normalize_audio_files():
    """
    Normalizes audio files to peak amplitude while preserving directory structure.
    Handles stereo/mono files and prevents division by zero in silent files.
    """
    # Configuration (consider using English paths)
    input_folder = r'\audio_without_silence'
    output_folder = r'\normalized_audio'
    target_peak = 0.95  # -1 dBFS headroom
    output_format = 'PCM_16'  # 16-bit WAV format

    os.makedirs(output_folder, exist_ok=True)

    # Process files with error handling
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith('.wav'):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)
            
            try:
                # Load audio with original properties
                y, sr = librosa.load(input_path, sr=None, mono=False)
                
                # Handle multi-channel audio
                if y.ndim == 1:
                    y = y.reshape(1, -1)  # Convert to 2D array for consistency
                    
                # Check for silent audio
                max_amp = np.max(np.abs(y))
                if max_amp == 0:
                    print(f"Warning: Silent file detected - {file_name}")
                    continue
                
                # Peak normalization with headroom
                y_normalized = librosa.util.normalize(
                    y, 
                    norm=np.inf, 
                    axis=1
                ) * target_peak
                
                # Save with specified format
                sf.write(
                    output_path,
                    y_normalized.T,  # Transpose for soundfile format
                    sr,
                    subtype=output_format
                )
                print(f"Normalized: {file_name}")
                
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")

if __name__ == "__main__":
    normalize_audio_files()
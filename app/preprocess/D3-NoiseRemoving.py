import os
import librosa  # Audio loading and processing
import noisereduce as nr  # Noise reduction library
import soundfile as sf  # Audio file I/O

def remove_noise(input_file, output_file):
    """
    Reduce noise from audio file using spectral gating.
    
    Args:
        input_file (str): Path to input audio file
        output_file (str): Path to save cleaned audio
    """
    try:
        # Load audio with original sampling rate
        y, sr = librosa.load(input_file, sr=None)
        
        # Perform noise reduction using default settings
        # (Automatically estimates noise from non-speech portions)
        reduced_noise = nr.reduce_noise(
            y=y, 
            sr=sr,
            stationary=False,  # For non-stationary noise
            prop_decrease=0.95  # 95% noise reduction
        )
        
        # Save as 16-bit PCM WAV format
        sf.write(output_file, reduced_noise, sr, subtype='PCM_16')
        print(f"Success: {input_file} -> {output_file}")
    
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")

def process_audio_files_in_folder(input_folder, output_folder):
    """
    Process all WAV files in directory structure while preserving folder hierarchy.
    
    Args:
        input_folder (str): Root directory with audio files
        output_folder (str): Target directory for cleaned files
    """
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(".wav"):  # Case-insensitive check
                input_path = os.path.join(root, file)
                
                # Maintain directory structure in output
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                
                output_path = os.path.join(output_dir, file)
                remove_noise(input_path, output_path)

# Configuration (recommended to use English paths)
input_folder = r'\Normalized_Audio'
output_folder = "Denoised_Audio"

# Execute processing
if __name__ == "__main__":
    process_audio_files_in_folder(input_folder, output_folder)
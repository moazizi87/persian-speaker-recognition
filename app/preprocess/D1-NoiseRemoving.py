# Import required libraries
import librosa  # For audio processing and analysis
import numpy as np  # For numerical operations and array handling
import os  # For file system operations
import soundfile as sf  # For writing audio files

def remove_noise_from_audio():
    """
    This function processes WAV files in a folder, applies basic noise reduction,
    and saves cleaned versions to a new folder.
    The noise reduction method uses a simple amplitude thresholding technique.
    """
    
    # Define input and output paths
    input_folder = r'\myaudio_tiny\myaudio'  # Path to raw audio files
    output_folder = r'\myaudio_tiny_cleaned'  # Path for cleaned files
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process each file in input folder
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith('.wav'):  # Check for WAV files (case-insensitive)
            
            # Construct full file paths
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)

            # Load audio file
            # y: audio time series (numpy array)
            # sr: sampling rate (default 22050 Hz)
            y, sr = librosa.load(input_path, sr=None)  # sr=None preserves original sampling rate

            # Noise estimation (first 0.5 seconds assumed to be noise)
            noise_samples = y[0:int(0.5 * sr)]  # Extract first 0.5 seconds
            noise_threshold = np.mean(np.abs(noise_samples))  # Calculate average amplitude

            # Apply noise reduction
            # Simple thresholding: keep samples above noise level, zero others
            cleaned_audio = np.where(np.abs(y) > noise_threshold, y, 0)

            # Save processed audio
            # soundfile preserves original audio format and sampling rate
            sf.write(output_path, cleaned_audio, sr, subtype='PCM_16')  # Save as 16-bit PCM WAV

# Execute the function when script is run directly
if __name__ == "__main__":
    remove_noise_from_audio()
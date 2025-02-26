import os
import numpy as np
import librosa  # Audio processing library
import soundfile as sf  # Audio file I/O

def remove_silence_from_audio():
    """
    Removes silent segments from WAV audio files in a directory.
    Uses librosa's voice activity detection to identify non-silent regions.
    Preserves original sampling rate and audio format.
    """
    
    # Input/Output configuration
    input_folder = r'\myaudio_tiny\myaudio'  # Raw audio files
    output_folder = r'\audio_without_silence'  # Processed files
    os.makedirs(output_folder, exist_ok=True)  # Create output directory if needed

    # Process each file in input directory
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith('.wav'):  # Case-insensitive WAV check
            try:
                # Construct file paths
                input_path = os.path.join(input_folder, file_name)
                output_path = os.path.join(output_folder, file_name)

                # Load audio with native sampling rate
                audio, sr = librosa.load(input_path, sr=None)

                # Detect non-silent intervals (20dB threshold)
                # Higher top_db values preserve more audio
                non_silent_intervals = librosa.effects.split(audio, top_db=20)

                # Combine non-silent segments
                processed_audio = np.concatenate([
                    audio[start:end] 
                    for start, end in non_silent_intervals
                ])

                # Save processed audio
                sf.write(output_path, processed_audio, sr, subtype='PCM_16')
                print(f"Processed: {file_name}")

            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")

# Execute when run directly
if __name__ == "__main__":
    remove_silence_from_audio()
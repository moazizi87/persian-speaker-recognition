import os
import librosa  # For audio processing and silence removal
import numpy as np
import soundfile as sf  # For writing audio files

def remove_silence(input_file, output_file, silence_thresh=-40, min_silence_len=500):
    """
    Remove silent segments from an audio file using amplitude threshold detection.
    
    Args:
        input_file (str): Path to input audio file
        output_file (str): Path to save processed audio
        silence_thresh (int): Silence threshold in decibels (negative values)
        min_silence_len (int): [Not implemented] Minimum silence duration to remove (ms)
    """
    try:
        # Load audio with native sampling rate
        y, sr = librosa.load(input_file, sr=None)
        
        # Detect non-silent intervals (librosa uses top_db for threshold)
        # Note: min_silence_len is not used in librosa's implementation
        intervals = librosa.effects.split(
            y, 
            top_db=-silence_thresh  # Convert negative threshold to positive dB
        )
        
        # Concatenate non-silent intervals
        non_silent_audio = np.concatenate([y[start:end] for start, end in intervals])
        
        # Save processed audio
        sf.write(output_file, non_silent_audio, sr)
        print(f"Processed: {input_file} -> Saved to: {output_file}")
    
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")

def process_audio_files_in_folder(input_folder, output_folder, silence_thresh=-40, min_silence_len=500):
    """
    Process all WAV files in a folder and its subfolders, preserving directory structure.
    
    Args:
        input_folder (str): Root directory containing audio files
        output_folder (str): Target directory for processed files
        silence_thresh (int): Silence detection threshold
        min_silence_len (int): Minimum silence duration to remove
    """
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(".wav"):
                # Build input path
                input_file_path = os.path.join(root, file)
                
                # Create mirrored output directory structure
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                
                # Build output path
                output_file_path = os.path.join(output_dir, file)
                
                # Process and save audio file
                remove_silence(
                    input_file_path, 
                    output_file_path, 
                    silence_thresh, 
                    min_silence_len
                )

# Configuration
input_folder = r'\SampleDeepMine\wav'
output_folder = "SilenceRemoved"  # English folder name recommended

# Execute processing
if __name__ == "__main__":
    process_audio_files_in_folder(input_folder, output_folder)
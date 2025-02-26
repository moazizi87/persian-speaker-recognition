# Import required libraries
# librosa for audio processing and analysis
import librosa
# os for interacting with the operating system and file handling
import os

def check_audio_durations_in_folder():
    """
    This function calculates and prints the duration of all WAV audio files
    in a specified folder using librosa library.
    """
    
    # Define the path to the folder containing audio files
    # NOTE: Replace this path with your actual folder path
    folder_path = r'\myaudio_tiny\myaudio'
    
    # Iterate through all files in the specified folder
    for file_name in os.listdir(folder_path):
        # Check if the file has a .wav extension
        if file_name.lower().endswith('.wav'):
            # Create full file path by joining folder path and file name
            file_path = os.path.join(folder_path, file_name)
            
            # Load audio file using librosa
            # y: audio time series (numpy array)
            # sr: sampling rate of the audio file
            y, sr = librosa.load(file_path, sr=None)  # sr=None preserves original sampling rate
            
            # Calculate audio duration in seconds
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Print formatted duration with 2 decimal places
            print(f"Duration of {file_name}: {duration:.2f} seconds")

# Execute the function when the script is run
if __name__ == "__main__":
    check_audio_durations_in_folder()
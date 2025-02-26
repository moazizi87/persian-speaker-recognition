import os
from pydub import AudioSegment  # For audio normalization

def normalize_audio(input_file, output_file):
    """
    Normalize audio file using peak normalization to -0.1 dBFS.
    
    Args:
        input_file (str): Path to input audio file
        output_file (str): Path to save normalized audio
    """
    try:
        # Load audio file using pydub
        audio = AudioSegment.from_file(input_file)
        
        # Perform peak normalization (default: -0.1 dBFS)
        normalized_audio = audio.normalize()
        
        # Export as 16-bit PCM WAV format
        normalized_audio.export(output_file, format="wav", parameters=["-ac", "1", "-ar", "16000"])
        print(f"Processed: {input_file} -> Saved to: {output_file}")
    
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")

def process_audio_files_in_folder(input_folder, output_folder):
    """
    Process all WAV files in a folder hierarchy, preserving directory structure.
    
    Args:
        input_folder (str): Root directory containing audio files
        output_folder (str): Target directory for normalized files
    """
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(".wav"):  # Case-insensitive check
                input_path = os.path.join(root, file)
                
                # Preserve directory structure in output
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                
                output_path = os.path.join(output_dir, file)
                
                normalize_audio(input_path, output_path)

# Configuration (consider using English path names)
input_folder = r'\SilenceRemoved\Dataset2'
output_folder = "Normalized_Audio"

# Execute processing
if __name__ == "__main__":
    process_audio_files_in_folder(input_folder, output_folder)
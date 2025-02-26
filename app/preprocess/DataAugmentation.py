# Import required libraries
import os  # File system operations
import librosa  # Audio processing and analysis
import numpy as np  # Numerical computing
import soundfile as sf  # Audio file I/O
from scipy.signal import butter, lfilter  # Digital filtering
from sklearn.preprocessing import StandardScaler  # Feature normalization

# Define input/output paths
input_folder = "/Audio/No Label"
output_folder = "../../data"
os.makedirs(output_folder, exist_ok=True)  # Create output directory if not exists

def extract_features(audio, sr, n_mfcc=13):
    """
    Extract MFCC features from audio signal.
    
    Args:
        audio (np.array): Input audio signal
        sr (int): Sampling rate
        n_mfcc (int): Number of MFCC coefficients to extract
    
    Returns:
        np.array: Concatenated array of MFCC means and standard deviations
    """
    # Compute MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    # Calculate statistics across time frames
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    return np.concatenate((mfccs_mean, mfccs_std))

def add_noise(audio, noise_level=0.005):
    """
    Add Gaussian noise to audio signal.
    
    Args:
        audio (np.array): Input audio signal
        noise_level (float): Standard deviation of noise distribution
    
    Returns:
        np.array: Noisy audio signal
    """
    noise = np.random.normal(0, noise_level, audio.shape)
    return audio + noise

def change_speed(audio, sr, speed_factor=1.1):
    """
    Change audio playback speed without affecting pitch.
    
    Args:
        audio (np.array): Input audio signal
        sr (int): Original sampling rate
        speed_factor (float): Speed modification factor (>1 speeds up)
    
    Returns:
        np.array: Time-stretched audio
    """
    return librosa.effects.time_stretch(audio, rate=speed_factor)

def change_pitch(audio, sr, n_steps=2):
    """
    Shift audio pitch by specified number of semitones.
    
    Args:
        audio (np.array): Input audio signal
        sr (int): Original sampling rate
        n_steps (int): Number of semitones to shift
    
    Returns:
        np.array: Pitch-shifted audio
    """
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def bandpass_filter(audio, sr, lowcut=500, highcut=4000, order=5):
    """
    Apply Butterworth bandpass filter to audio signal.
    
    Args:
        audio (np.array): Input audio signal
        sr (int): Sampling rate
        lowcut (int): Low cutoff frequency (Hz)
        highcut (int): High cutoff frequency (Hz)
        order (int): Filter order
    
    Returns:
        np.array: Filtered audio signal
    """
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, audio)

def reverse_audio(audio):
    """Reverse the audio signal temporally."""
    return audio[::-1]

def augment_audio(audio, sr):
    """
    Generate multiple augmented versions of input audio.
    
    Args:
        audio (np.array): Input audio signal
        sr (int): Sampling rate
    
    Returns:
        list: List of augmented audio versions
    """
    augmented_versions = []
    # Original audio with additive noise
    augmented_versions.append(add_noise(audio)) 
    # Speed modified version
    augmented_versions.append(change_speed(audio, sr, speed_factor=1.1))
    # Pitch shifted version
    augmented_versions.append(change_pitch(audio, sr, n_steps=-2))
    # Bandpass filtered version
    augmented_versions.append(bandpass_filter(audio, sr))
    # Temporally reversed version
    augmented_versions.append(reverse_audio(audio))
    return augmented_versions

# Initialize storage for features and filenames
features_list = []  # Stores extracted MFCC features
file_names = []  # Stores original filenames

# Process each audio file in input directory
for file_name in os.listdir(input_folder):
    if file_name.lower().endswith('.wav'):
        input_path = os.path.join(input_folder, file_name)
        
        # Load audio file with original sampling rate
        audio, sr = librosa.load(input_path, sr=None)
        
        # Generate augmented audio versions
        augmented_audios = augment_audio(audio, sr)
        
        # Extract features for each augmented version
        for augmented_audio in augmented_audios:
            features = extract_features(augmented_audio, sr)
            features_list.append(features)
            file_names.append(file_name)

# Save augmented audio files
for i, file_name in enumerate(file_names):
    # Create unique output filename
    base_name = os.path.splitext(file_name)[0]
    output_path = os.path.join(output_folder, f"{base_name}_augmented_{i}.wav")
    
    # Regenerate augmented audio (note: this could be optimized)
    original_audio, sr = librosa.load(os.path.join(input_folder, file_name), sr=None)
    augmented_versions = augment_audio(original_audio, sr)
    augmented_audio = augmented_versions[i % 5]  # Cycle through 5 augmentation types
    
    # Save as 16-bit PCM WAV file
    sf.write(output_path, augmented_audio, sr, subtype='PCM_16')

print("Data augmentation completed successfully!")
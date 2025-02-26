import streamlit as st
import os
import numpy as np
import torch
import librosa
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import csv
import ast
from pydub import AudioSegment
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Define the enhanced model
class EnhancedTransformerClassifier(torch.nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=256, num_heads=8, num_layers=4, dropout=0.3):
        super(EnhancedTransformerClassifier, self).__init__()
        self.conv1d = torch.nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm1d(hidden_size)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        self.embedding = torch.nn.Linear(hidden_size, hidden_size)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4, dropout=dropout, batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = torch.nn.Linear(hidden_size // 2, num_classes)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the pre-trained model
def load_model(model_save_path):
    try:
        model = EnhancedTransformerClassifier(input_size=128, num_classes=15).to(device)
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        model.eval()
        logging.info("Model loaded successfully.")
        return model
    except Exception as error:
        logging.error(f"Error loading model: {error}")
        return None

# Load mean and standard deviation for normalization
def load_normalization():
    try:
        mean = np.load("mean_mel.npy")
        std = np.load("std_mel.npy")
        if mean.shape != (128,):
            mean = np.mean(mean, axis=0)
        if std.shape != (128,):
            std = np.mean(std, axis=0)
        return mean, std
    except Exception as error:
        logging.error(f"Error loading normalization files: {error}")
        return None, None

# Function to convert audio to WAV format
def convert_to_wav(audio_path):
    try:
        audio = AudioSegment.from_file(audio_path)
        temp_wav_path = os.path.splitext(audio_path)[0] + "_converted.wav"
        audio.export(temp_wav_path, format="wav")
        return temp_wav_path
    except Exception as error:
        logging.error(f"Error converting {audio_path} to WAV: {error}")
        return None

# Function to extract Mel-Spectrogram features
def extract_mel_spectrogram(audio_path, n_mels=128):
    try:
        if not audio_path.lower().endswith(".wav"):
            logging.info(f"Converting {audio_path} to WAV format...")
            audio_path = convert_to_wav(audio_path)
            if audio_path is None:
                return None

        y, sr = librosa.load(audio_path, sr=None)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return mel_spectrogram_db.T
    except Exception as error:
        logging.error(f"Error extracting Mel-Spectrogram from {audio_path}: {error}")
        return None

# Preprocess audio file
def preprocess_audio(audio_path, mean, std):
    try:
        logging.info(f"Preprocessing audio: {audio_path}")
        mel_spectrogram = extract_mel_spectrogram(audio_path)
        if mel_spectrogram is None:
            return None
        mel_spectrogram_norm = (mel_spectrogram - mean) / std
        return torch.tensor(mel_spectrogram_norm, dtype=torch.float32).unsqueeze(0).to(device)
    except Exception as error:
        logging.error(f"Error preprocessing audio: {error}")
        return None

# Extract embeddings using the model
def extract_embeddings(audio_path, model, mean, std):
    try:
        logging.info(f"Extracting embeddings for: {audio_path}")
        mel_spectrogram_tensor = preprocess_audio(audio_path, mean, std)
        if mel_spectrogram_tensor is None:
            return None
        with torch.no_grad():
            embeddings = model(mel_spectrogram_tensor)
        embeddings = embeddings.cpu().numpy().flatten()
        embeddings = embeddings / np.linalg.norm(embeddings)
        return embeddings
    except Exception as error:
        logging.error(f"Error extracting embeddings: {error}")
        return None

# Load speaker database
def load_speaker_database(csv_file):
    try:
        if not os.path.exists(csv_file):
            return {}
        
        speaker_database = {}
        with open(csv_file, mode="r") as file:
            reader = csv.reader(file)
            for row in reader:
                speaker_name = row[0]
                age = int(row[1])
                sex = row[2]
                embeddings = [np.array(ast.literal_eval(embedding), dtype=np.float32) for embedding in row[3:]]
                speaker_database[speaker_name] = {"embeddings": embeddings, "age": age, "sex": sex}
        return speaker_database
    except Exception as error:
        logging.error(f"Error loading speaker database: {error}")
        return None

# Save speaker database
def save_speaker_database(speaker_database, csv_file):
    try:
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            for speaker_name, data in speaker_database.items():
                embeddings_str = [str(embedding.tolist()) for embedding in data["embeddings"]]
                writer.writerow([speaker_name, data["age"], data["sex"]] + embeddings_str)
    except Exception as error:
        logging.error(f"Error saving speaker database: {error}")

# Enroll a new speaker
def enroll_speaker(audio_path, speaker_name, age, sex, model, mean, std, csv_file):
    try:
        if not os.path.exists(audio_path):
            logging.error(f"Audio file not found: {audio_path}")
            return
        
        logging.info(f"Enrolling speaker: {speaker_name}")
        
        embeddings = extract_embeddings(audio_path, model, mean, std)
        if embeddings is None:
            logging.error("No embeddings extracted.")
            return
        
        speaker_database = load_speaker_database(csv_file)
        if speaker_database is None:
            speaker_database = {}
        
        if speaker_name not in speaker_database:
            speaker_database[speaker_name] = {"embeddings": [], "age": age, "sex": sex}
        speaker_database[speaker_name]["embeddings"].append(embeddings)
        
        save_speaker_database(speaker_database, csv_file)
        logging.info(f"Speaker '{speaker_name}' enrolled successfully.")
    except Exception as error:
        logging.error(f"Error enrolling speaker: {error}")

# Recognize a speaker
def recognize_speaker(audio_path, model, mean, std, csv_file, threshold=0.3):
    try:
        if not os.path.exists(audio_path):
            logging.error(f"Audio file not found: {audio_path}")
            return None, None
        
        logging.info(f"Recognizing speaker from: {audio_path}")
        embeddings = extract_embeddings(audio_path, model, mean, std)
        if embeddings is None:
            return None, None
        
        speaker_database = load_speaker_database(csv_file)
        if speaker_database is None:
            return None, None
        
        best_match = None
        best_score = -1

        for name, data in speaker_database.items():
            enrolled_embeddings_list = data["embeddings"]
            total_similarity = 0.0
            for enrolled_embeddings in enrolled_embeddings_list:
                score = cosine_similarity([embeddings], [enrolled_embeddings])[0][0]
                total_similarity += score
            
            avg_similarity = total_similarity / len(enrolled_embeddings_list)
            logging.info(f"Average similarity with {name}: {avg_similarity:.4f}")
            
            if avg_similarity > best_score:
                best_score = avg_similarity
                best_match = name

        if best_score >= threshold:
            logging.info(f"Recognized speaker: {best_match} (Average Score: {best_score:.4f})")
            return best_match, best_score
        else:
            logging.info(f"Speaker not recognized. Best average score: {best_score:.4f}")
            return None, best_score
    except Exception as error:
        logging.error(f"Error recognizing speaker: {error}")
        return None, None

# Visualize speaker embeddings
def visualize_embeddings(csv_file):
    try:
        speaker_database = load_speaker_database(csv_file)
        if not speaker_database:
            logging.error("Speaker database is empty.")
            return
        
        names = list(speaker_database.keys())
        embeddings = np.array([embedding for data in speaker_database.values() for embedding in data["embeddings"]])
        
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        for i, name in enumerate(names):
            plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], label=name)
        plt.legend()
        plt.title("Speaker Embeddings Visualization")
        st.pyplot(plt)
    except Exception as error:
        logging.error(f"Error visualizing embeddings: {error}")

# Streamlit app
def main():
    st.title("Speaker Recognition System")
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Choose an action", ["Enroll Speaker", "Recognize Speaker", "Visualize Embeddings"])

    model = load_model("enhanced_transformer_embedding.pth")
    mean, std = load_normalization()
    if model is None or mean is None or std is None:
        st.error("Failed to load model or normalization files.")
        return

    csv_file = "speaker_database.csv"

    if choice == "Enroll Speaker":
        st.header("Enroll a New Speaker")
        audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
        speaker_name = st.text_input("Enter speaker name")
        age = st.number_input("Enter speaker age", min_value=0, max_value=120, value=30)
        sex = st.selectbox("Enter speaker sex", ["Male", "Female", "Other"])
        if st.button("Enroll"):
            if audio_file is not None and speaker_name:
                with open(os.path.join("test_audio", audio_file.name), "wb") as f:
                    f.write(audio_file.getbuffer())
                enroll_speaker(os.path.join("test_audio", audio_file.name), speaker_name, age, sex, model, mean, std, csv_file)
                st.success(f"Speaker '{speaker_name}' enrolled successfully.")
            else:
                st.error("Please upload an audio file and enter a speaker name.")

    elif choice == "Recognize Speaker":
        st.header("Recognize a Speaker")
        audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
        if st.button("Recognize"):
            if audio_file is not None:
                with open(os.path.join("test_audio", audio_file.name), "wb") as f:
                    f.write(audio_file.getbuffer())
                recognized_speaker, score = recognize_speaker(os.path.join("test_audio", audio_file.name), model, mean, std, csv_file)
                if recognized_speaker:
                    speaker_info = load_speaker_database(csv_file).get(recognized_speaker, {})
                    age = speaker_info.get("age", "Unknown")
                    sex = speaker_info.get("sex", "Unknown")
                    st.success(f"Recognized speaker: {recognized_speaker} (Age: {age}, Sex: {sex}, Average Score: {score:.4f})")
                else:
                    st.error(f"Speaker not recognized. Best average score: {score:.4f}")
            else:
                st.error("Please upload an audio file.")

    elif choice == "Visualize Embeddings":
        st.header("Visualize Speaker Embeddings")
        if st.button("Show Visualization"):
            visualize_embeddings(csv_file)

if __name__ == "__main__":
    main()
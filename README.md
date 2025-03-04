# Persian Speaker Recognition Project

This repository hosts a complete speaker recognition system that leverages advanced audio preprocessing, feature extraction, and deep learning techniques. The project integrates an interactive web interface built with Streamlit, and its entire environment is containerized using Docker for hassle-free deployment.

Model Performance Report
========================

Device used: cuda

Cross-Validation Results:</br>
Cross-Validation Accuracies: [0.9390060240963856, 0.9363704819277109, 0.9307228915662651, 0.9371234939759037, 0.9352409638554217] </br>
Mean Cross-Validation Accuracy: 0.9357 </br>
Cross-Validation F1-Scores: [0.9380916877555588, 0.9358488653700947, 0.9304451060787098, 0.9367799343916686, 0.9351866344576445] </br>
Mean Cross-Validation F1-Score: 0.9353 </br>

---

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Preprocessing](#preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Deployment](#deployment)
- [Installation and Usage](#installation-and-usage)
- [Docker Support](#docker-support)
- [Team](#Team)
- [License](#license)
- [Contact](#contact)

---

## Overview
Due to time and resource constraints, creating a new dataset from scratch was deemed impractical. Instead, three existing datasets were evaluated for their quality and diversity. After thorough review, only two datasets met the stringent quality standards required for this project. The system includes:
- A robust preprocessing pipeline tailored for two different datasets.
- Feature extraction routines that compute statistical properties (mean and standard deviation of Mel features) in both NumPy and PyTorch formats.
- A final model script that encapsulates training and inference logic.
- A Streamlit-based web application for interactive evaluation.

---

## Project Structure
```
speaker-recognition/
├── Dockerfile
├── app/
│   ├── extracted features/
│   │   ├── mean_mel.npy
│   │   ├── mean_mel.pt
│   │   ├── std_mel.npy
│   │   └── std_mel.pt
│   ├── final_model.py
│   ├── preprocess/
│   │   ├── D1-GetDuration.py
│   │   ├── D1-NoiseRemoving.py
│   │   ├── D1-Normalization.py
│   │   ├── D1-SilenceRemoving.py
│   │   ├── D3-NoiseRemoving.py
│   │   ├── D3-PydubNormalization.py
│   │   ├── D3-SilenceRemoving.py
│   │   └── DataAugmentation.py
│   └── streamlit.py
├── data/
│   ├── 1000_augmented_0.wav
│   ├── ...
│   ├── 9_augmented_4.wav
│   └── augmented_labels.csv
├── models/
│   ├── model_performance_report.txt
│   └── enhanced_transformer_embedding.pth
└── requirements.txt
```


### Explanation of Key Directories and Files:
- **Dockerfile**: Contains instructions to build the Docker image for containerizing the project.
- **app/**: The main application code.
  - **extracted features/**: Stores precomputed Mel feature statistics (mean and standard deviation) in both NumPy (`.npy`) and PyTorch (`.pt`) formats.
  - **final_model.py**: Implements the final model architecture and includes training/inference routines.
  - **preprocess/**: Contains multiple scripts for audio preprocessing.  
    - Files prefixed with `D1-` correspond to operations applied to **Dataset 1**.
    - Files prefixed with `D3-` correspond to operations applied to **Dataset 3**.
    - **DataAugmentation.py**: Implements various data augmentation techniques to enhance model robustness.
  - **streamlit.py**: The entry point for launching the interactive Streamlit web interface.
- **data/**: Contains augmented audio csv file, called `augmented_labels.csv` which holds corresponding labels. The augmented audio dataset is available at the link in the next section.
- **models/**: Contains model performance reports and the saved model weights.
- **requirements.txt**: Lists all Python dependencies needed to run the project.

---

## Datasets
Three datasets were initially considered for this project. However, after extensive evaluation based on audio quality, diversity of speakers, and recording conditions, only two were selected for training and evaluation.

1. **Dataset 1**:
   - **Description**: High-quality recordings with diverse speakers and optimal recording conditions.
   - **Duration**: Approximately 3 hours of audio.
   - **Usage**: Selected for both training and evaluation due to its superior quality.
   - **Access**: [Data.DeepMine](https://data.deepmine.ir)

2. **Dataset 2 (YouTube Data) - We just checked this and it was not applicable to our project.**:
   - **Description**: A large (approx. 8GB) dataset from YouTube featuring diverse audio samples.
   - **Challenges**: Contains significant noise, persistent background music, and extended periods of silence.
   - **Outcome**: Discarded due to poor alignment with the project’s quality requirements.
   - **Access**: [Kaggle Dataset](https://www.kaggle.com/datasets/amirpourmand/automatic-speech-recognition-farsi-youtube/data?select=audios)

3. **Dataset 3**:
   - **Description**: Consists of recordings with acceptable quality and a diverse range of speakers.
   - **Duration**: Approximately 3 hours of audio.
   - **Usage**: Selected for training and testing alongside Dataset 1.
   - **Access**: [Persian Speech Dataset on GitHub](https://github.com/persiandataset/PersianSpeech/tree/main)

**Final Dataset Combination**:  
After combining Dataset 1 and Dataset 3, the total audio duration was initially 6 hours. With additional data augmentation techniques (detailed in the preprocessing section), the effective dataset was expanded to approximately 12 hours across 7036 audio files.
Link to access our gained and augmented dataset in Google Drive: https://drive.google.com/file/d/13ew8bzz0Zm1xhbJp3XbuMVS5qgHtOp0V/view?usp=sharing

---

## Preprocessing
The preprocessing pipeline is designed to handle the specific characteristics of the selected datasets. The scripts in the `app/preprocess/` directory perform the following operations:

- **D1-GetDuration.py**: Calculates the duration of each audio file in Dataset 1.
- **D1-NoiseRemoving.py**: Removes unwanted background noise from Dataset 1 recordings.
- **D1-Normalization.py**: Normalizes the amplitude levels of Dataset 1 audio.
- **D1-SilenceRemoving.py**: Detects and removes silence in Dataset 1 recordings.
- **D3-NoiseRemoving.py**: Applies noise reduction specifically tuned for Dataset 3.
- **D3-PydubNormalization.py**: Uses the Pydub library to normalize audio levels in Dataset 3.
- **D3-SilenceRemoving.py**: Eliminates silent segments in Dataset 3 audio.
- **DataAugmentation.py**: Implements various augmentation techniques (e.g., pitch shifting, time stretching) to artificially expand the dataset and improve model generalization.

*Note:* The file naming convention (prefix `D1-` and `D3-`) clearly indicates which preprocessing steps correspond to which dataset.

---

## Model Training and Evaluation
- **final_model.py**: This script encompasses the entire model training pipeline. It:
  - Loads and preprocesses the extracted audio features.
  - Trains the speaker recognition model.
  - Evaluates model performance using validation data.
  - Saves the trained model weights and generates a performance report.
  
- **models/model_performance_report.txt**: Contains detailed metrics and evaluation results from the training phase.
- **models/enhanced_transformer_embedding.pth**: The serialized model weights from the final training session.

---

## Deployment
The project features an interactive web interface built with Streamlit, allowing users to:
- Upload and process audio files.
- View real-time predictions from the speaker recognition model.
- Explore data preprocessing and model inference outcomes interactively.

### Running the Streamlit App Locally:
1. **Install Dependencies** (see [Installation and Usage](#installation-and-usage) below).
2. **Launch the App**:
   ```bash
   streamlit run app/streamlit.py
   ```
3. **Access the app at http://localhost:8501.**

---

## Installation and Usage

### Prerequisites
- **Python 3.9+**
- **Docker** (if using containerization)

### Local Installation
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/YourUsername/speaker-recognition.git
   cd speaker-recognition
   ```
2. **Install Python Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Preprocessing Scripts (Do it if you want to do these preprocesses on your own audio files.):**
   Execute the necessary scripts in `app/preprocess/` to prepare your data. For example:
   ```bash
   python app/preprocess/D1-NoiseRemoving.py
   ```
   *(Repeat as needed for each preprocessing step.)*

4. **Train or Evaluate the Model (Do it if you want to do this training phase on your own data.):**
   ```bash
   python app/final_model.py
   ```

5. **Launch the Web Application:**
   ```bash
   streamlit run app/streamlit.py
   ```

### Using Docker
1. **Build the Docker Image:**
   ```bash
   docker build -t persian-speaker-recognition .
   ```
2. **Run the Docker Container:**
   ```bash
   docker run -p 8501:8501 persian-speaker-recognition
   ```
   Access the app at `http://localhost:8501`.

---

## Docker Support
The provided `Dockerfile` ensures that all dependencies and configurations are encapsulated within a container. This allows users to run the project on any system with Docker installed, without manually setting up Python environments or dependencies.

**Key Dockerfile Highlights:**
- Uses the official `python:3.9-slim` image as a base.
- Copies the application code and installs dependencies via `pip install -r requirements.txt`.
- Sets the working directory and launches the Streamlit application with the command:
  ```bash
  CMD ["streamlit", "run", "app/streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
  ```

---

## Team
Mohammad Omid Azizi, Zeinab Torabi, Samaneh Hashemian and Zahra Marami

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Contact
For any questions, suggestions, or contributions, please open an issue on GitHub or contact [mohammadomid1387@gmail.com].

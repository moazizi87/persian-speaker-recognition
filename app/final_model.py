import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Load preprocessed data
features_file = "/extracted_mel_features_augmented.npy"
labels_file = "/labels_augmented.npy"

features_array = np.load(features_file)
labels_array = np.load(labels_file)

# Prepare data for the model
X = torch.tensor(features_array, dtype=torch.float32)
y = torch.tensor(labels_array, dtype=torch.long)

# Normalize features
mean = X.mean(dim=(0, 1), keepdim=True)
std = X.std(dim=(0, 1), keepdim=True)
X = (X - mean) / std

# Save mean and std for later use
torch.save(mean, "extracted features/mean_mel.pt")
torch.save(std, "extracted features/std_mel.pt")
logging.info("Mean and std saved to mean_mel.pt and std_mel.pt")

# Enhanced Transformer-based model for embedding generation
class EnhancedTransformerEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_heads=8, num_layers=4, dropout=0.3, num_classes=None):
        super(EnhancedTransformerEmbedding, self).__init__()
        
        # 1D Convolutional layer to extract local features
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Transformer Encoder
        self.embedding = nn.Linear(hidden_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Fully connected layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes) if num_classes else None
    
    def forward(self, x):
        # Input shape: (batch, sequence_length, input_size)
        x = x.permute(0, 2, 1)  # (batch, input_size, sequence_length)
        x = self.conv1d(x)  # (batch, hidden_size, sequence_length)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)  # (batch, hidden_size, sequence_length // 2)
        x = x.permute(0, 2, 1)  # (batch, sequence_length // 2, hidden_size)
        
        # Transformer
        x = self.embedding(x)  # (batch, sequence_length // 2, hidden_size)
        x = x.permute(1, 0, 2)  # (sequence_length // 2, batch, hidden_size)
        x = self.transformer(x)  # (sequence_length // 2, batch, hidden_size)
        x = x.mean(dim=0)  # Global average pooling
        
        # Fully connected layers
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        if self.fc2:
            x = self.fc2(x)  # Classification head
        return x

# Define model parameters
input_size = features_array.shape[2]  # Number of Mel bands
hidden_size = 256
num_classes = len(np.unique(labels_array))

# Cross-validation setup
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Lists to store cross-validation results
cv_train_losses = []
cv_val_losses = []
cv_accuracies = []
cv_f1_scores = []

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=5, delta=0.01):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Training and evaluation loop
for fold, (train_ids, val_ids) in enumerate(kfold.split(X)):
    print(f"Fold {fold + 1}/{k_folds}")
    
    # Split data into training and validation sets
    X_train, X_val = X[train_ids], X[val_ids]
    y_train, y_val = y[train_ids], y[val_ids]
    
    # Create TensorDataset and DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=True)
    
    # Instantiate model, loss function, and optimizer
    model = EnhancedTransformerEmbedding(input_size=input_size, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=5, delta=0.01)
    
    # Lists to store losses for plotting
    train_losses = []
    val_losses = []
    
    # Train the model
    epochs = 30
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        # Step the scheduler
        scheduler.step(val_loss)
        
        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]}")
    
    # Store losses for this fold
    cv_train_losses.append(train_losses)
    cv_val_losses.append(val_losses)
    
    # Evaluate the model on the validation set
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    
    # Calculate accuracy and F1-score for this fold
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    cv_accuracies.append(accuracy)
    cv_f1_scores.append(f1)
    print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

# Print cross-validation results
print(f"Cross-Validation Accuracies: {cv_accuracies}")
print(f"Mean Cross-Validation Accuracy: {np.mean(cv_accuracies):.4f}")
print(f"Cross-Validation F1-Scores: {cv_f1_scores}")
print(f"Mean Cross-Validation F1-Score: {np.mean(cv_f1_scores):.4f}")

# Plot learning curves for the last fold
plt.figure(figsize=(10, 5))
plt.plot(cv_train_losses[-1], label='Training Loss')
plt.plot(cv_val_losses[-1], label='Validation Loss')
plt.title('Learning Curves for Last Fold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("learning_curves.png")  # Save the plot as an image
plt.show()

# Save the model
model_save_path = "../models/enhanced_transformer_embedding.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Save the report to a file
report_filename = "../models/model_performance_report.txt"
with open(report_filename, "w") as report_file:
    report_file.write("Model Performance Report\n")
    report_file.write("========================\n\n")
    report_file.write(f"Device used: {device}\n\n")
    report_file.write("Cross-Validation Results:\n")
    report_file.write(f"Cross-Validation Accuracies: {cv_accuracies}\n")
    report_file.write(f"Mean Cross-Validation Accuracy: {np.mean(cv_accuracies):.4f}\n")
    report_file.write(f"Cross-Validation F1-Scores: {cv_f1_scores}\n")
    report_file.write(f"Mean Cross-Validation F1-Score: {np.mean(cv_f1_scores):.4f}\n\n")
    report_file.write("Learning curves for the last fold have been saved as 'learning_curves.png'.\n")
    report_file.write(f"Model saved to {model_save_path}\n")

print(f"Report saved to {report_filename}")
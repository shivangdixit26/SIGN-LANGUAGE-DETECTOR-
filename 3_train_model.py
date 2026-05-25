"""
3_train_model.py
================
Trains a lightweight Dense Neural Network for sign language recognition.

Uses PyTorch (compatible with Python 3.14).
Loads hand landmark features from keypoints.csv and trains a DNN model
optimized for real-time edge inference.

Input:  keypoints.csv (from 2_extract_features.py)
Output: models/sign_model.pt (PyTorch model)
        models/label_encoder.pkl (label mapping)

Model Architecture:
- Input: 63 features (21 hand landmarks × 3 coordinates)
- Dense(128) + ReLU + Dropout(0.3)
- Dense(64) + ReLU + Dropout(0.3)  
- Dense(32) + ReLU + Dropout(0.2)
- Dense(36) + Softmax (output for 0-9, A-Z)

Usage:
    python 3_train_model.py
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
INPUT_FILE = Path("keypoints.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "sign_model.pt"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
SCALER_PATH = MODEL_DIR / "feature_scaler.pkl"

# Training hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
BATCH_SIZE = 16  # Smaller batches for better gradient estimates
EPOCHS = 100  # Standard training
LEARNING_RATE = 0.001  # Standard learning rate
EARLY_STOPPING_PATIENCE = 15
NUM_FEATURES = 63  # 21 landmarks × 3 coords
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_SMOOTHING = 0.05
AUGMENT_REPEATS = 2


def normalize_landmarks(flat_landmarks):
    """Normalize one 63-d landmark vector to be translation/scale invariant."""
    pts = np.asarray(flat_landmarks, dtype=np.float32).reshape(21, 3)

    # Translation invariance: center on wrist (landmark 0)
    pts = pts - pts[0]

    # Scale invariance: normalize by hand extent in XY plane
    xy_norms = np.linalg.norm(pts[:, :2], axis=1)
    scale = float(np.max(xy_norms))
    if scale < 1e-6:
        scale = 1.0

    pts = pts / scale
    return pts.flatten().astype(np.float32)


def preprocess_features(X):
    """Apply landmark normalization to all samples."""
    return np.stack([normalize_landmarks(row) for row in X], axis=0).astype(np.float32)


def augment_landmarks(X, y, repeats=AUGMENT_REPEATS):
    """Augment normalized landmarks with small geometric/noise perturbations."""
    X_list = [X]
    y_list = [y]

    for _ in range(repeats):
        augmented = []
        for sample in X:
            pts = sample.reshape(21, 3).copy()

            # Random in-plane rotation (camera/view variation)
            theta = np.deg2rad(np.random.uniform(-12.0, 12.0))
            c, s = np.cos(theta), np.sin(theta)
            rot = np.array([[c, -s], [s, c]], dtype=np.float32)
            pts[:, :2] = pts[:, :2] @ rot.T

            # Random scale jitter
            scale = np.random.uniform(0.90, 1.10)
            pts[:, :2] *= scale

            # Small additive noise
            pts[:, :2] += np.random.normal(0.0, 0.01, size=(21, 2)).astype(np.float32)
            pts[:, 2] += np.random.normal(0.0, 0.005, size=(21,)).astype(np.float32)

            augmented.append(pts.flatten())

        X_aug = np.asarray(augmented, dtype=np.float32)
        X_list.append(X_aug)
        y_list.append(y.copy())

    X_out = np.vstack(X_list).astype(np.float32)
    y_out = np.hstack(y_list)
    logger.info(f"Augmentation applied: {len(X)} -> {len(X_out)} samples")
    return X_out, y_out


class SignLanguageNet(nn.Module):
    """Lightweight DNN for sign language recognition."""
    
    def __init__(self, num_classes=36, input_features=NUM_FEATURES):
        super(SignLanguageNet, self).__init__()
        
        # Network layers - simpler architecture for stability
        self.fc1 = nn.Linear(input_features, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(0.2)
        self.relu3 = nn.ReLU()
        
        self.fc4 = nn.Linear(32, num_classes)
        
        # Regularization
        self.l2_lambda = 1e-5
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        return x
    
    def get_l2_loss(self):
        """L2 regularization loss."""
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param)
        return self.l2_lambda * l2_loss


def load_data(filepath):
    """Load and preprocess data."""
    logger.info(f"Loading data from {filepath}...")
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    # Load CSV - label is the first column
    df = pd.read_csv(filepath)
    logger.info(f"✓ Loaded {len(df)} samples")
    
    # Separate labels and features
    # Label column is first, features are the rest
    y = df.iloc[:, 0].values.astype(str)  # First column is label
    X = df.iloc[:, 1:].values.astype(np.float32)  # Rest are features
    X = preprocess_features(X)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    logger.info(f"  Features shape: {X.shape}")
    logger.info(f"  Labels: {sorted(label_encoder.classes_)}")
    class_counts = np.bincount(y_encoded)
    logger.info(f"  Class distribution: Min={class_counts.min()} | Avg={class_counts.mean():.1f} | Max={class_counts.max()}")
    
    return X, y_encoded, label_encoder, class_counts


def train_model(X_train, y_train, X_val, y_val, num_classes):
    """Train the model."""
    logger.info("\n" + "="*60)
    logger.info("Training Model")
    logger.info("="*60)
    
    # Create model
    model = SignLanguageNet(num_classes=num_classes).to(DEVICE)
    logger.info(f"Model moved to device: {DEVICE}")
    
    # Calculate class weights to handle imbalance - simple approach
    class_counts = np.bincount(y_train)
    class_weights = len(class_counts) / (class_counts.astype(np.float32) + 1.0)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)
    
    logger.info(f"Class weights: Min={class_weights.min():.2f} | Max={class_weights.max():.2f}")
    
    # Loss function with class weights and optimizer - NO scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Prepare data
    X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
    y_train_tensor = torch.LongTensor(y_train).to(DEVICE)
    X_val_tensor = torch.FloatTensor(X_val).to(DEVICE)
    y_val_tensor = torch.LongTensor(y_val).to(DEVICE)
    
    # Simple DataLoader with shuffling - NO weighted sampler
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Training loop
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    logger.info(f"Training for up to {EPOCHS} epochs (early stopping: {EARLY_STOPPING_PATIENCE})...\n")
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            l2_loss = model.get_l2_loss()
            total_loss = loss + l2_loss
            
            total_loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        
        train_loss /= len(X_train)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            val_pred = torch.argmax(val_outputs, dim=1)
            val_accuracy = (val_pred == y_val_tensor).float().mean().item()
        
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        scheduler.step()
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1:3d}/{EPOCHS} | "
                       f"Train Loss: {train_loss:.4f} | "
                       f"Val Loss: {val_loss:.4f} | "
                       f"Val Acc: {val_accuracy:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logger.info(f"\n⏹ Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break
    
    return model, train_losses, val_losses, val_accuracies


def evaluate_model(model, X_test, y_test, label_encoder):
    """Evaluate model on test set."""
    logger.info("\n" + "="*60)
    logger.info("Model Evaluation")
    logger.info("="*60)
    
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
    y_test_tensor = torch.LongTensor(y_test).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        accuracy = (predictions == y_test).mean()
    
    logger.info(f"\n✓ Overall Test Accuracy: {accuracy:.4f}")
    
    # Per-class accuracy
    logger.info("\nPer-class metrics:")
    logger.info("-" * 50)
    class_accuracies = {}
    low_performers = []
    
    for i, label in enumerate(label_encoder.classes_):
        mask = y_test == i
        if mask.sum() > 0:
            class_acc = (predictions[mask] == i).mean()
            class_count = mask.sum()
            class_accuracies[label] = class_acc
            
            # Flag low-performing classes
            if class_acc < 0.5:
                low_performers.append((label, class_acc, class_count))
            
            status = "✓" if class_acc >= 0.8 else "⚠" if class_acc >= 0.5 else "✗"
            logger.info(f"{status} {label:3s}: Acc={class_acc:.4f} | Samples={class_count}")
    
    if low_performers:
        logger.info("\n⚠ Low-performing classes (accuracy < 50%):")
        for label, acc, count in sorted(low_performers, key=lambda x: x[1]):
            logger.info(f"   {label}: {acc:.4f} accuracy ({count} samples)")
    
    return accuracy, class_accuracies


def save_model(model, label_encoder, model_path, label_path):
    """Save model and label encoder."""
    logger.info(f"\n⏳ Saving model to {model_path}...")
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save PyTorch model
    torch.save(model.state_dict(), model_path)
    logger.info(f"✓ Model saved: {model_path}")
    
    # Save label encoder
    joblib.dump(label_encoder, label_path)
    logger.info(f"✓ Label encoder saved: {label_path}")
    
    logger.info(f"\nModel size: {model_path.stat().st_size / 1024:.1f} KB")


def save_scaler(scaler, scaler_path):
    """Save fitted feature scaler."""
    joblib.dump(scaler, scaler_path)
    logger.info(f"✓ Feature scaler saved: {scaler_path}")


def main():
    """Main training pipeline."""
    logger.info("\n" + "="*60)
    logger.info("Sign Language DNN Training (PyTorch)")
    logger.info("="*60)
    
    # Load data
    X, y, label_encoder, class_counts = load_data(INPUT_FILE)
    
    # Split data
    logger.info(f"\nSplitting data...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    val_split = VALIDATION_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_split,
        random_state=RANDOM_STATE, stratify=y_train_val
    )
    
    logger.info(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Augment training split only (after split, before scaling)
    X_train, y_train = augment_landmarks(X_train, y_train)

    # Fit scaler on train only, then transform val/test
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    
    # Train model
    num_classes = len(label_encoder.classes_)
    model, train_losses, val_losses, val_accs = train_model(
        X_train, y_train, X_val, y_val, num_classes
    )
    
    # Evaluate
    test_accuracy, class_accs = evaluate_model(model, X_test, y_test, label_encoder)
    
    # Save model
    save_model(model, label_encoder, MODEL_PATH, LABEL_ENCODER_PATH)
    save_scaler(scaler, SCALER_PATH)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("✓ Training Complete!")
    logger.info("="*60)
    logger.info(f"  Test Accuracy: {test_accuracy:.2%}")
    logger.info(f"  Model: {MODEL_PATH}")
    logger.info(f"  Encoder: {LABEL_ENCODER_PATH}")
    logger.info(f"  Scaler: {SCALER_PATH}")
    logger.info(f"  Next step: Run 4_test_local_webcam.py")
    logger.info("="*60 + "\n")


if __name__ == "__main__":
    main()

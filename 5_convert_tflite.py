"""
5_convert_tflite.py / 5_quantize_model.py
==========================================
Converts trained PyTorch model to quantized format for edge deployment.

Loads the sign_model.pt, applies dynamic quantization,
and saves as model_quantized.pt for Raspberry Pi deployment.

Quantization benefits:
- Reduced model size (50-75% smaller)
- Faster inference on ARM devices
- Lower memory requirements
- Suitable for real-time edge processing

Input:  models/sign_model.pt
Output: models/model_quantized.pt

Usage:
    python 5_convert_tflite.py
"""

import torch
import torch.nn as nn
from pathlib import Path
import logging
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_DIR = Path("models")
PYTORCH_MODEL_PATH = MODEL_DIR / "sign_model.pt"
QUANTIZED_MODEL_PATH = MODEL_DIR / "model_quantized.pt"
QUANTIZED_STATE_PATH = MODEL_DIR / "model_quantized_state.pt"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
DEVICE = torch.device("cpu")  # Use CPU for quantization


class SignLanguageNet(nn.Module):
    """Lightweight DNN for sign language recognition."""
    
    def __init__(self, num_classes=36, input_features=63):
        super(SignLanguageNet, self).__init__()
        
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


def quantize_model(model_path, output_path):
    """
    Quantize the PyTorch model using dynamic quantization.
    
    Dynamic quantization:
    - Reduces weights to int8
    - Keeps activations in float32
    - No retraining required
    - Good for inference on CPU/ARM
    
    Args:
        model_path (Path): Path to trained model
        output_path (Path): Path to save quantized model
    """
    logger.info(f"Loading model from {model_path}...")
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load label encoder to get number of classes
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    num_classes = len(label_encoder.classes_)
    
    # Create and load model
    model = SignLanguageNet(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
    model.eval()
    
    logger.info(f"Model loaded. Number of classes: {num_classes}")
    
    # Get original model size
    original_size = model_path.stat().st_size / (1024 * 1024)
    logger.info(f"Original model size: {original_size:.2f} MB")
    
    # Count original parameters
    original_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Original parameters: {original_params:,}")
    
    # Apply dynamic quantization
    logger.info("\n⏳ Applying dynamic quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},  # Quantize Linear layers
        dtype=torch.qint8
    )
    
    logger.info("✓ Quantization complete")
    
    # Save quantized model directly (as a whole model, not state dict)
    logger.info(f"\nSaving quantized model to {output_path}...")
    torch.save(quantized_model, output_path)  # Save whole model, not state dict

    # Also save state_dict to avoid pickle incompatibility across devices
    logger.info(f"Saving quantized state_dict to {QUANTIZED_STATE_PATH}...")
    torch.save(quantized_model.state_dict(), QUANTIZED_STATE_PATH)
    
    # Compare sizes
    quantized_size = output_path.stat().st_size / (1024 * 1024)
    size_reduction = (1 - quantized_size / original_size) * 100
    
    logger.info("="*60)
    logger.info("Quantization Results")
    logger.info("="*60)
    logger.info(f"Original size:   {original_size:.2f} MB")
    logger.info(f"Quantized size:  {quantized_size:.2f} MB")
    logger.info(f"Size reduction:  {size_reduction:.1f}%")
    logger.info(f"Compression ratio: {original_size/quantized_size:.2f}x")
    logger.info("="*60)
    
    return True


def test_quantized_model(model_path, quantized_path):
    """
    Test quantized model with dummy input to verify it works.
    
    Args:
        model_path (Path): Original model path
        quantized_path (Path): Quantized model path
    """
    logger.info("\n" + "="*60)
    logger.info("Testing Quantized Model")
    logger.info("="*60)
    
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    num_classes = len(label_encoder.classes_)
    
    # Load original model
    logger.info("Loading original model...")
    original_model = SignLanguageNet(num_classes=num_classes).to(DEVICE)
    original_model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
    original_model.eval()
    
    # Load quantized model directly (it's already quantized)
    logger.info("Loading quantized model...")
    quantized_model = torch.load(quantized_path, map_location=DEVICE, weights_only=False)
    quantized_model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 63).to(DEVICE)
    
    # Run inference
    logger.info("\nRunning inference...")
    with torch.no_grad():
        original_output = original_model(dummy_input)
        quantized_output = quantized_model(dummy_input)
    
    # Compare outputs
    original_pred = torch.argmax(original_output, dim=1).item()
    quantized_pred = torch.argmax(quantized_output, dim=1).item()
    
    logger.info(f"Original prediction:  {label_encoder.classes_[original_pred]}")
    logger.info(f"Quantized prediction: {label_encoder.classes_[quantized_pred]}")
    logger.info(f"Match: {'✓ Yes' if original_pred == quantized_pred else '✗ No'}")
    
    # Measure inference speed
    import time
    
    logger.info("\nBenchmarking...")
    num_iterations = 100
    
    # Original model speed
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            original_model(dummy_input)
    original_time = (time.time() - start) / num_iterations * 1000
    
    # Quantized model speed
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            quantized_model(dummy_input)
    quantized_time = (time.time() - start) / num_iterations * 1000
    
    speedup = original_time / quantized_time if quantized_time > 0 else 1.0
    
    logger.info(f"Original inference: {original_time:.2f} ms")
    logger.info(f"Quantized inference: {quantized_time:.2f} ms")
    logger.info(f"Speedup: {speedup:.2f}x")
    
    logger.info("="*60)


def main():
    """Main entry point."""
    logger.info("\n" + "="*60)
    logger.info("PyTorch Model Quantization")
    logger.info("="*60)
    
    try:
        # Check if original model exists
        if not PYTORCH_MODEL_PATH.exists():
            logger.error(f"Model not found: {PYTORCH_MODEL_PATH}")
            logger.error("Please run 3_train_model.py first")
            return False
        
        if not LABEL_ENCODER_PATH.exists():
            logger.error(f"Label encoder not found: {LABEL_ENCODER_PATH}")
            return False
        
        # Quantize model
        MODEL_DIR.mkdir(exist_ok=True)
        quantize_model(PYTORCH_MODEL_PATH, QUANTIZED_MODEL_PATH)
        
        # Test quantized model
        test_quantized_model(PYTORCH_MODEL_PATH, QUANTIZED_MODEL_PATH)
        
        logger.info("\n" + "="*60)
        logger.info("✓ Quantization completed successfully!")
        logger.info(f"  Quantized model: {QUANTIZED_MODEL_PATH}")
        logger.info(f"  Quantized state: {QUANTIZED_STATE_PATH}")
        logger.info(f"  Next step: Run 6_pi_inference.py for deployment")
        logger.info("="*60 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"\n✗ Quantization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

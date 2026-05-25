# Quick Start Guide

## One-Line Summary
Download ASL dataset → Extract hand landmarks → Train lightweight DNN → Test locally → Quantize to TFLite → Deploy on Raspberry Pi

---

## Executive Timeline

### PC Setup (First Time Only)
**Time: ~20 minutes**

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements_local.txt

# 3. Setup Kaggle API
# Download kaggle.json from https://www.kaggle.com/settings/account
# Place in C:\Users\{YourUsername}\.kaggle\kaggle.json
```

### Full Pipeline (First Run)
**Total Time: ~2-3 hours**

```bash
# 1. Download dataset (~30 min, 2GB download)
python 1_fetch_data.py

# 2. Extract features (~30 min, uses MediaPipe)
python 2_extract_features.py

# 3. Train model (~10-15 min)
python 3_train_model.py

# 4. Test with webcam (~5 min, interactive)
python 4_test_local_webcam.py
# Press 'q' to quit

# 5. Convert to TFLite (~1 min)
python 5_convert_tflite.py

# 6. (On Raspberry Pi) Deploy and test
# See "Raspberry Pi Deployment" section below
```

---

## Subsequent Runs

### Just Testing Model (No Retraining)
```bash
python 4_test_local_webcam.py
```

### Key Files Created
- `keypoints.csv` - Features (100+ MB, can be deleted after training)
- `models/sign_model.h5` - Trained model (200 KB)
- `models/label_encoder.npy` - Class labels (< 1 KB)
- `models/model.tflite` - Quantized model (15 KB)

---

## Raspberry Pi Deployment

### Prerequisites
- Raspberry Pi 4B or later with Raspbian OS
- SSH access to Pi
- USB webcam

### Setup (One-time)
```bash
# On Pi, create project directory
ssh pi@raspberry.local
mkdir ~/sign_language
cd ~/sign_language

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Copy requirements from PC and install
# (Or manually: pip install tflite-runtime mediapipe opencv-python numpy)
pip install -r requirements_pi.txt
```

### Copy Models from PC
```powershell
# On PC PowerShell
scp models\model.tflite pi@raspberry.local:~/sign_language/
scp models\label_encoder.npy pi@raspberry.local:~/sign_language/
scp 6_pi_inference.py pi@raspberry.local:~/sign_language/
```

### Run on Pi
```bash
# On Pi
cd ~/sign_language
python 6_pi_inference.py
```

---

## Customization Checklist

### Want Better Accuracy?
1. Train on more diverse images
2. Increase EPOCHS in 3_train_model.py (default: 100)
3. Decrease LEARNING_RATE to 0.0005
4. Add augmentation to dataset

### Want Higher FPS on Pi?
1. Reduce FRAME_WIDTH/HEIGHT in 6_pi_inference.py (640x480 → 480x360)
2. Decrease SMOOTHING_WINDOW (5 → 2)
3. Disable MediaPipe drawing (comment out mp_drawing code)

### Want to Add New Gestures?
1. Add images to data/asl_alphabet/{NEW_LABEL}/
2. Re-run 2_extract_features.py
3. Re-run 3_train_model.py
4. Continue through 5_convert_tflite.py

---

## Troubleshooting Quick Links

| Problem | Solution |
|---------|----------|
| "Kaggle API not found" | Setup kaggle.json (see README.md) |
| "No hands detected" | Check lighting, clear hand gestures |
| "Low accuracy" | Check dataset quality, retrain with different LR |
| "Slow on Pi" | Reduce resolution, decrease smoothing |
| "Model file not found" | Run all scripts in order (1→6) |

---

## Performance Expectations

### Accuracy
- ASL Alphabet: 85-90%
- Depends on lighting, hand size, gesture clarity

### Speed
- **PC**: 30-60 FPS (CPU), 100+ FPS (GPU)
- **Raspberry Pi 4B**: 25-33 FPS (decent lighting)
- **Raspberry Pi 5**: 40+ FPS

### Model Sizes
- Keras model: ~200 KB
- TFLite model: ~15 KB (after quantization)
- Total deployment: <2 MB with dependencies

---

## File References

| File | Purpose | Input | Output |
|------|---------|-------|--------|
| 1_fetch_data.py | Download ASL dataset from Kaggle | - | data/asl_alphabet/ |
| 2_extract_features.py | MediaPipe landmarks extraction | data/asl_alphabet/ | keypoints.csv |
| 3_train_model.py | Train DNN on landmarks | keypoints.csv | models/sign_model.h5 |
| 4_test_local_webcam.py | Live testing on PC | models/sign_model.h5 | Video display |
| 5_convert_tflite.py | Quantize for edge | models/sign_model.h5 | models/model.tflite |
| 6_pi_inference.py | Deploy on Pi | models/model.tflite | Video display |

---

## Key Concepts

### MediaPipe Hand Landmarks
- 21 points per hand (knuckles, joints, etc.)
- 3D coordinates (x, y, z)
- Normalized to [0, 1]
- Flattened to 63-element array

### Lightweight DNN
- Input: 63 (hand landmarks)
- Hidden: 128 → 64 → 32 neurons
- Output: 26 or more (A-Z + others)
- ~10K parameters, <100ms inference

### TensorFlow Lite Quantization
- Compresses float32 → int8 weights
- Keeps activations as float32
- 4-8x smaller, 1-5% accuracy loss
- Optimized for ARM devices

---

## Pro Tips

### For Best Accuracy
1. Use consistent lighting
2. Keep hand 0.5-1.5m from camera
3. Show clear, complete hand gesture
4. Use validation set to monitor overfitting

### For Best Performance
1. Lower SMOOTHING_WINDOW for real-time responsiveness
2. Increase BATCH_SIZE for faster training
3. Use GPU if available (10x faster)
4. Cache predictions with temporal smoothing

### For Production Deployment
1. Test on multiple Pi units
2. Monitor inference latency (track inference_time)
3. Log predictions for offline analysis
4. Add confidence threshold filtering
5. Handle edge cases (partial hands, multiple hands)

---

## Next Steps

- [ ] Read full README.md for comprehensive documentation
- [ ] Run complete pipeline (1_fetch_data.py → 6_pi_inference.py)
- [ ] Fine-tune hyperparameters based on your needs
- [ ] Deploy to Raspberry Pi
- [ ] Test with real ASL users if possible

---

**Good luck! The complete pipeline is set up and ready to run. 🚀**

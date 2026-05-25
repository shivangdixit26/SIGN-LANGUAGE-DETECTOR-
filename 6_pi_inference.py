"""
6_pi_inference.py (PyTorch Version)
===================================
Real-time sign language recognition on Raspberry Pi using PyTorch.

Optimized for edge inference on constrained hardware:
- Uses lightweight PyTorch quantized model
- Reduced precision (int8 weights)
- CPU-only inference
- Optimized for 25-30+ FPS on Raspberry Pi 4B

Input:  model_quantized.pt and label_encoder.pkl (from PC)
Output: Live video feed with predictions

Deployment on Raspberry Pi:
1. Copy model_quantized.pt and label_encoder.pkl to Pi
2. Install requirements: pip install torch torchvision mediapipe opencv-python joblib -q
3. Run: python 6_pi_inference.py

Controls:
- Press 'q' to quit
- Press 's' to save a screenshot
- Press 'p' to pause/resume

Usage:
    python 6_pi_inference.py
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import logging
from collections import deque
from datetime import datetime
import time
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = Path("model_quantized.pt")
MODEL_STATE_PATH = Path("model_quantized_state.pt")
LABEL_ENCODER_PATH = Path("label_encoder.pkl")
CAMERA_ID = 0
FRAME_WIDTH = 640  # Lower resolution for Pi
FRAME_HEIGHT = 480
FPS_DISPLAY_INTERVAL = 30
CONFIDENCE_THRESHOLD = 0.6
SMOOTHING_WINDOW = 3  # Prediction voting window
NUM_FEATURES = 63
DEVICE = torch.device("cpu")  # Always CPU for Pi

# Display configuration
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_COLOR = (0, 255, 0)  # BGR - Green
FONT_COLOR_RED = (0, 0, 255)  # BGR - Red
FONT_THICKNESS = 2
BBOX_COLOR = (0, 255, 0)  # Green bounding box
BBOX_THICKNESS = 2


class SignLanguageNet(nn.Module):
    """Lightweight DNN for sign language recognition."""
    
    def __init__(self, num_classes=36, input_features=NUM_FEATURES):
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


class RaspberryPiSignDetector:
    """Optimized sign language detector for Raspberry Pi."""
    
    def __init__(self, model_path, label_encoder_path):
        """Initialize the detector."""
        # Load model
        logger.info(f"Loading quantized model from {model_path}...")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load label encoder to get number of classes
        self.label_encoder = joblib.load(label_encoder_path)
        num_classes = len(self.label_encoder.classes_)
        
        # Create and load model
        base_model = SignLanguageNet(num_classes=num_classes).to(DEVICE)
        try:
            loaded = torch.load(model_path, map_location=DEVICE, weights_only=False)
            if isinstance(loaded, dict):
                base_model.load_state_dict(loaded)
                self.model = base_model
            else:
                self.model = loaded
            logger.info(f"✓ Quantized model loaded ({model_path.stat().st_size / 1024:.1f} KB)")
        except Exception as exc:
            if MODEL_STATE_PATH.exists():
                state_dict = torch.load(MODEL_STATE_PATH, map_location=DEVICE, weights_only=False)
                # Recreate quantized module structure so packed params match.
                quantized_model = torch.quantization.quantize_dynamic(
                    base_model, {nn.Linear}, dtype=torch.qint8
                )
                quantized_model.load_state_dict(state_dict)
                self.model = quantized_model
                logger.info(f"✓ Quantized state_dict loaded ({MODEL_STATE_PATH.stat().st_size / 1024:.1f} KB)")
            else:
                raise RuntimeError(
                    "Failed to load quantized model. Re-export model_quantized_state.pt "
                    "using 5_convert_tflite.py and copy it to the Pi."
                ) from exc

        self.model.eval()
        
        # Initialize MediaPipe Hand Landmarker
        self._init_mediapipe()
        
        # Metrics
        self.frame_count = 0
        self.inference_times = deque(maxlen=30)
        self.fps = 0
        self.prediction_history = deque(maxlen=SMOOTHING_WINDOW)
        self.no_hand_frames = 0
    
    def _init_mediapipe(self):
        """Initialize MediaPipe Hand Landmarker with Vision Tasks API."""
        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python import BaseOptions
            import urllib.request
            
            # Get or download hand landmarker model
            cache_dir = Path.home() / ".cache" / "mediapipe" / "vision"
            cache_dir.mkdir(parents=True, exist_ok=True)
            model_file = cache_dir / "hand_landmarker.task"
            
            if not model_file.exists():
                model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
                logger.info("Downloading MediaPipe hand landmarker model...")
                urllib.request.urlretrieve(model_url, model_file)
            
            # Create hand landmarker
            base_options = BaseOptions(model_asset_path=str(model_file))
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
            logger.info("✓ MediaPipe Hand Landmarker initialized")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            raise
    
    def extract_landmarks(self, frame):
        """Extract hand landmarks from frame."""
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb.astype(np.uint8)
            )
            
            # Detect hand landmarks
            detection_result = self.hand_landmarker.detect(mp_image)
            
            # Check if hand detected
            if not detection_result.hand_landmarks:
                return None, False, None
            
            # Extract landmarks
            hand_landmarks = detection_result.hand_landmarks[0]
            landmarks = np.array([
                [lm.x, lm.y, lm.z]
                for lm in hand_landmarks
            ]).flatten().astype(np.float32)
            
            return landmarks, True, detection_result
            
        except Exception as e:
            logger.error(f"Error extracting landmarks: {e}")
            return None, False, None
    
    @staticmethod
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

    def predict(self, landmarks):
        """Run inference with quantized model."""
        try:
            start_time = time.time()
            
            # Normalize features
            norm_landmarks = self.normalize_landmarks(landmarks)
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(norm_landmarks).unsqueeze(0).to(DEVICE)
            
            # Run inference
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence = probabilities.max().item()
                predicted_idx = torch.argmax(output, dim=1).item()
            
            inference_time = (time.time() - start_time) * 1000
            self.inference_times.append(inference_time)
            
            # Get label
            predicted_label = self.label_encoder.classes_[predicted_idx]
            
            return predicted_label, float(confidence)
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None, 0.0
    
    def get_smoothed_prediction(self):
        """Get prediction with temporal smoothing."""
        if not self.prediction_history:
            return None, 0.0
        
        labels_voted = [pred[0] for pred in self.prediction_history]
        if labels_voted:
            smoothed_label = max(set(labels_voted), key=labels_voted.count)
            avg_confidence = np.mean([pred[1] for pred in self.prediction_history])
            return smoothed_label, avg_confidence
        
        return None, 0.0
    
    def draw_info(self, frame, fps, landmark_status, prediction, confidence, inference_ms):
        """Draw information on frame."""
        frame_copy = frame.copy()
        
        # FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame_copy, fps_text, (10, 30),
                   FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
        
        # Inference time
        if self.inference_times:
            avg_inf_time = np.mean(list(self.inference_times))
            inf_text = f"Inference: {avg_inf_time:.1f}ms"
            cv2.putText(frame_copy, inf_text, (10, 60),
                       FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
        
        # Landmark status
        status_color = FONT_COLOR if "detected" in landmark_status else FONT_COLOR_RED
        cv2.putText(frame_copy, landmark_status, (10, 90),
                   FONT, FONT_SCALE, status_color, FONT_THICKNESS)
        
        # Prediction
        if prediction is not None and confidence > CONFIDENCE_THRESHOLD:
            pred_text = f"Predicted: {prediction} ({confidence*100:.0f}%)"
            cv2.putText(frame_copy, pred_text, (10, 120),
                       FONT, 1.0, FONT_COLOR, FONT_THICKNESS)
        
        # Frame counter
        frame_text = f"Frame: {self.frame_count}"
        cv2.putText(frame_copy, frame_text, (10, 150),
                   FONT, 0.6, (255, 255, 255), 1)
        
        return frame_copy
    
    def draw_landmarks(self, frame, detection_result):
        """Draw hand landmarks on frame."""
        try:
            if not detection_result or not detection_result.hand_landmarks:
                return frame
            
            h, w = frame.shape[:2]
            
            for hand_landmarks in detection_result.hand_landmarks:
                # Draw landmarks
                for i, landmark in enumerate(hand_landmarks):
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                
                # Draw connections
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (0, 9), (9, 10), (10, 11), (11, 12),
                    (0, 13), (13, 14), (14, 15), (15, 16),
                    (0, 17), (17, 18), (18, 19), (19, 20),
                ]
                
                for start, end in connections:
                    x1 = int(hand_landmarks[start].x * w)
                    y1 = int(hand_landmarks[start].y * h)
                    x2 = int(hand_landmarks[end].x * w)
                    y2 = int(hand_landmarks[end].y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            return frame
        except Exception as e:
            logger.error(f"Error drawing landmarks: {e}")
            return frame
    
    def run(self, camera_id=0):
        """Run real-time detection loop optimized for Pi."""
        logger.info(f"Opening camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error(f"Cannot open camera {camera_id}")
            return False
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for low latency
        
        logger.info(f"Starting Pi inference. Press 'q' to quit, 's' to save, 'p' to pause.")
        
        frame_times = deque(maxlen=FPS_DISPLAY_INTERVAL)
        paused = False
        pause_frame = None
        
        try:
            while True:
                if not paused:
                    start_time = time.time()
                    ret, frame = cap.read()
                    
                    if not ret:
                        logger.error("Failed to read frame")
                        break
                    
                    # Flip frame horizontally to act like a mirror
                    frame = cv2.flip(frame, 1)
                    
                    self.frame_count += 1
                    
                    # Extract landmarks
                    landmarks, hand_found, detection_result = self.extract_landmarks(frame)
                    
                    # Make prediction
                    if hand_found and landmarks is not None:
                        predicted_label, confidence = self.predict(landmarks)
                        if predicted_label is not None:
                            self.prediction_history.append((predicted_label, confidence))
                        self.no_hand_frames = 0
                        landmark_status = "✓ Hand detected"
                    else:
                        self.no_hand_frames += 1
                        landmark_status = "✗ No hand"
                    
                    # Get smoothed prediction
                    smoothed_label, smoothed_conf = self.get_smoothed_prediction()
                    
                    # Calculate FPS
                    frame_times.append(time.time() - start_time)
                    if len(frame_times) == FPS_DISPLAY_INTERVAL:
                        self.fps = FPS_DISPLAY_INTERVAL / sum(frame_times)
                    
                    # Draw
                    display_frame = self.draw_info(
                        frame, self.fps, landmark_status, 
                        smoothed_label, smoothed_conf,
                        np.mean(list(self.inference_times)) if self.inference_times else 0
                    )
                    
                    # Draw landmarks
                    if detection_result:
                        display_frame = self.draw_landmarks(display_frame, detection_result)
                    
                    pause_frame = display_frame
                else:
                    if pause_frame is not None:
                        display_frame = pause_frame.copy()
                        cv2.putText(display_frame, "PAUSED", (FRAME_WIDTH//2 - 80, 60),
                                   FONT, 1.5, (0, 0, 255), 3)
                
                # Display
                cv2.imshow("Pi Sign Language Interpreter", display_frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit signal received")
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"pi_screenshot_{timestamp}.png"
                    cv2.imwrite(filename, display_frame)
                    logger.info(f"Screenshot saved: {filename}")
                elif key == ord('p'):
                    paused = not paused
                    logger.info(f"{'Paused' if paused else 'Resumed'}")
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Camera closed")
            logger.info(f"Total frames: {self.frame_count}")
            if self.frame_count > 0:
                logger.info(f"Frames without hand: {self.no_hand_frames} "
                           f"({self.no_hand_frames/self.frame_count*100:.1f}%)")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Raspberry Pi Sign Language Interpreter")
    print("=" * 60)
    
    try:
        if not MODEL_PATH.exists():
            logger.error(f"Model not found: {MODEL_PATH}")
            return False
        
        if not LABEL_ENCODER_PATH.exists():
            logger.error(f"Label encoder not found: {LABEL_ENCODER_PATH}")
            return False
        
        detector = RaspberryPiSignDetector(MODEL_PATH, LABEL_ENCODER_PATH)
        detector.run(CAMERA_ID)
        
        print("\n" + "=" * 60)
        print("✓ Pi inference completed!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

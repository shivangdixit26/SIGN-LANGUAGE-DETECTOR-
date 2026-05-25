"""
6_pi_inference_mobile.py (PyTorch Version - Mobile Webcam)
===========================================================
Real-time ASL recognition on Raspberry Pi using IP Webcam from mobile phone.

Supports:
- Android: IP Webcam app (free from Google Play Store)
- iOS: Foscam app
- Also works with USB webcam as fallback

Setup:
1. Copy model_quantized.pt and label_encoder.pkl to Pi
2. Install requirements: pip install torch torchvision mediapipe opencv-python joblib numpy
3. On your mobile phone, download IP Webcam app and start the server
4. Run this script with: python 6_pi_inference_mobile.py --camera http://192.168.x.x:8080

Controls:
- Press 'q' to quit
- Press 's' to save a screenshot
- Press 'p' to pause/resume
- Press 'f' to toggle FPS display
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
import argparse
import urllib.request
import urllib.error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = Path("model_quantized.pt")
LABEL_ENCODER_PATH = Path("label_encoder.pkl")
FRAME_WIDTH = 1280  # Full HD for display
FRAME_HEIGHT = 720
BOX_SIZE = 500      # Detection box dimension
BOX_X = FRAME_WIDTH - BOX_SIZE - 30  # Top right placement
BOX_Y = 30
FPS_DISPLAY_INTERVAL = 30
CONFIDENCE_THRESHOLD = 0.6
SMOOTHING_WINDOW = 3
NUM_FEATURES = 63
DEVICE = torch.device("cpu")

# Display configuration
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_COLOR = (0, 255, 0)
FONT_COLOR_RED = (0, 0, 255)
FONT_THICKNESS = 2
BBOX_COLOR = (0, 255, 0)
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
        x = self.relu1(self.dropout1(self.fc1(x)))
        x = self.relu2(self.dropout2(self.fc2(x)))
        x = self.relu3(self.dropout3(self.fc3(x)))
        x = self.fc4(x)
        return x


class MobileSignDetector:
    """Sign language detector optimized for mobile webcam streams."""
    
    def __init__(self, model_path, label_encoder_path):
        """Initialize the detector."""
        logger.info("Initializing Sign Language Detector...")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load label encoder
        self.label_encoder = joblib.load(label_encoder_path)
        num_classes = len(self.label_encoder.classes_)
        logger.info(f"Labels: {self.label_encoder.classes_}")
        
        # Load model
        self.model = torch.load(model_path, map_location=DEVICE, weights_only=False)
        self.model.eval()
        logger.info(f"✓ Model loaded ({model_path.stat().st_size / 1024:.1f} KB)")
        
        # Initialize MediaPipe
        self._init_mediapipe()
        
        # Metrics
        self.frame_count = 0
        self.inference_times = deque(maxlen=30)
        self.fps = 0
        self.prediction_history = deque(maxlen=SMOOTHING_WINDOW)
        self.no_hand_frames = 0
    
    def _init_mediapipe(self):
        """Initialize MediaPipe Hand Landmarker."""
        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.components import processors
            import mediapipe as mp_support
            
            # Use BaseOptions to specify the model
            base_options = mp_support.tasks.BaseOptions(
                model_asset_path=None  # Will use default bundled model
            )
            
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
            logger.warning(f"Using legacy MediaPipe: {e}")
            self.mp_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.use_legacy = True
    
    def extract_landmarks(self, frame, draw_frame=None):
        """Extract hand landmarks from frame."""
        try:
            if hasattr(self, 'hand_landmarker'):
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                detection_result = self.hand_landmarker.detect(mp_image)
                
                if detection_result.hand_landmarks and len(detection_result.hand_landmarks) > 0:
                    landmarks = detection_result.hand_landmarks[0]
                    
                    if draw_frame is not None:
                        from mediapipe.framework.formats import landmark_pb2
                        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                        hand_landmarks_proto.landmark.extend([
                            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks
                        ])
                        mp.solutions.drawing_utils.draw_landmarks(
                            draw_frame,
                            hand_landmarks_proto,
                            mp.solutions.hands.HAND_CONNECTIONS,
                            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                            mp.solutions.drawing_styles.get_default_hand_connections_style()
                        )
                        
                    features = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
                    return features if len(features) == NUM_FEATURES else None
            else:
                results = self.mp_hands.process(frame)
                if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
                    landmarks = results.multi_hand_landmarks[0]
                    
                    if draw_frame is not None:
                        mp.solutions.drawing_utils.draw_landmarks(
                            draw_frame,
                            landmarks,
                            mp.solutions.hands.HAND_CONNECTIONS,
                            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                            mp.solutions.drawing_styles.get_default_hand_connections_style()
                        )
                        
                    features = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
                    return features if len(features) == NUM_FEATURES else None
        except Exception as e:
            logger.debug(f"Landmark extraction error: {e}")
        
        return None
    
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

    def predict(self, features):
        """Predict sign from features."""
        # Normalize features
        norm_features = self.normalize_landmarks(features)
        
        with torch.no_grad():
            input_tensor = torch.FloatTensor(norm_features).unsqueeze(0).to(DEVICE)
            start_time = time.time()
            
            output = self.model(input_tensor)
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)
            
            return (
                self.label_encoder.classes_[predicted_idx.item()],
                confidence.item(),
                inference_time
            )
    
    def smooth_prediction(self, prediction):
        """Apply smoothing to predictions."""
        self.prediction_history.append(prediction[0])
        
        if len(self.prediction_history) >= SMOOTHING_WINDOW:
            # Return most common prediction
            from collections import Counter
            most_common = Counter(self.prediction_history).most_common(1)[0][0]
            return most_common
        
        return prediction[0]
    
    def run(self, camera_source):
        """Run the detector on video stream."""
        logger.info(f"Opening camera: {camera_source}")
        
        cap = cv2.VideoCapture(camera_source)
        
        # Try to set resolution for local cameras
        if isinstance(camera_source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        else:
            logger.info("Mobile webcam stream detected - auto-adjusting resolution")
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera: {camera_source}")
        
        logger.info("✓ Camera opened successfully")
        logger.info("\nControls:")
        logger.info("  'q' - Quit")
        logger.info("  's' - Save screenshot")
        logger.info("  'p' - Pause/Resume")
        logger.info("  'f' - Toggle FPS display")
        
        paused = False
        show_fps = True
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning("Failed to read frame")
                    continue
                
                # Flip frame horizontally to act like a mirror
                frame = cv2.flip(frame, 1)
                
                # Resize frame
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if not paused:
                    # Extract landmarks from the FULL frame for maximum MediaPipe stability
                    features = self.extract_landmarks(frame_rgb, draw_frame=frame)
                    
                    # Draw reference box purely as a visual UI guide
                    cv2.rectangle(frame, (BOX_X, BOX_Y), (BOX_X + BOX_SIZE, BOX_Y + BOX_SIZE), (255, 200, 0), 3)
                    cv2.putText(frame, "Keep Hand Here", (BOX_X + 10, BOX_Y + 30), FONT, 0.7, (255, 200, 0), 2)

                    if features is not None:
                        # Predict
                        prediction = self.predict(features)
                        label, confidence, inf_time = prediction
                        
                        # Smooth prediction
                        smooth_label = self.smooth_prediction(prediction)
                        
                        # Draw results
                        self.no_hand_frames = 0
                        confidence_pct = confidence * 100
                        
                        # Draw prediction below the box
                        text = f"Sign: {smooth_label}"
                        cv2.putText(frame, text, (BOX_X, BOX_Y + BOX_SIZE + 40), FONT, 1.2, FONT_COLOR, 3)
                        
                        # Color based on confidence
                        color = FONT_COLOR if confidence > CONFIDENCE_THRESHOLD else FONT_COLOR_RED
                        text_conf = f"Conf: {confidence_pct:.1f}%"
                        cv2.putText(frame, text_conf, (BOX_X, BOX_Y + BOX_SIZE + 80), FONT, 0.9, color, 2)
                    else:
                        self.no_hand_frames += 1
                        cv2.putText(frame, "Searching...", (BOX_X, BOX_Y + BOX_SIZE + 40), FONT, 1.0, FONT_COLOR_RED, 2)
                
                # Display FPS
                if show_fps:
                    if len(self.inference_times) > 0:
                        avg_inf_time = np.mean(self.inference_times)
                        fps = 1 / avg_inf_time if avg_inf_time > 0 else 0
                        fps_text = f"Inf: {avg_inf_time*1000:.1f}ms | FPS: {fps:.1f}"
                        cv2.putText(frame, fps_text, (FRAME_WIDTH - 350, FRAME_HEIGHT - 20), 
                                   FONT, FONT_SCALE-0.2, FONT_COLOR, 1)
                
                # Display paused status
                if paused:
                    cv2.putText(frame, "PAUSED (Press 'p' to resume)", (150, 240), 
                               FONT, 1.2, FONT_COLOR_RED, FONT_THICKNESS)
                
                # Show frame
                cv2.imshow("ASL Sign Language - Real-time Detection", frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quitting...")
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"Screenshot saved: {filename}")
                elif key == ord('p'):
                    paused = not paused
                    logger.info(f"{'Paused' if paused else 'Resumed'}")
                elif key == ord('f'):
                    show_fps = not show_fps
                    logger.info(f"FPS display: {'ON' if show_fps else 'OFF'}")
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Runtime error: {e}", exc_info=True)
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Camera closed")


def check_camera_connection(camera_source):
    """Check if camera source is available."""
    if isinstance(camera_source, str):
        logger.info(f"Testing connection to {camera_source}...")
        try:
            response = urllib.request.urlopen(camera_source + "?t=" + str(time.time()), timeout=5)
            logger.info(f"✓ Mobile webcam is accessible")
            return True
        except Exception as e:
            logger.error(f"✗ Cannot connect to mobile webcam: {e}")
            logger.info("Make sure:")
            logger.info("1. Mobile phone and Pi are on same WiFi network")
            logger.info("2. IP Webcam app is running on mobile")
            logger.info("3. You're using the correct IP address")
            return False
    else:
        logger.info(f"Testing connection to local camera (ID: {camera_source})...")
        cap = cv2.VideoCapture(camera_source)
        if cap.isOpened():
            logger.info(f"✓ Local camera is accessible")
            cap.release()
            return True
        else:
            logger.error(f"✗ Cannot access local camera")
            return False


def main():
    parser = argparse.ArgumentParser(
        description='Real-time ASL recognition with mobile or USB webcam',
        epilog='''
Examples:
  # Mobile webcam (IP Webcam app):
  python 6_pi_inference_mobile.py --camera http://192.168.1.10:8080

  # Local USB webcam:
  python 6_pi_inference_mobile.py --camera 0

  # Default (tries mobile first, falls back to USB):
  python 6_pi_inference_mobile.py
        '''
    )
    
    parser.add_argument(
        '--camera',
        type=str,
        default=None,
        help='Camera source: IP address (http://x.x.x.x:8080) or USB ID (0, 1, etc.)'
    )
    
    args = parser.parse_args()
    
    # Determine camera source
    camera_source = args.camera
    
    if camera_source is None:
        logger.info("No camera specified. Trying mobile webcam first...")
        logger.info("Make sure your IP Webcam app is running!")
        logger.info("\nTo use mobile webcam, run:")
        logger.info("  python 6_pi_inference_mobile.py --camera http://YOUR_PHONE_IP:8080")
        logger.info("\nFalling back to local USB camera (ID: 0)...")
        camera_source = 0
    
    # Convert string camera source to proper format
    if isinstance(camera_source, str) and not camera_source.startswith('http'):
        try:
            camera_source = int(camera_source)
        except ValueError:
            pass
    
    # Check camera connection
    if not check_camera_connection(camera_source):
        logger.error("Failed to connect to camera. Exiting.")
        return
    
    try:
        # Initialize detector
        detector = MobileSignDetector(MODEL_PATH, LABEL_ENCODER_PATH)
        
        # Run detector
        detector.run(camera_source)
        
    except FileNotFoundError as e:
        logger.error(f"Required file not found: {e}")
        logger.info("\nMake sure these files exist in current directory:")
        logger.info("  - model_quantized.pt")
        logger.info("  - label_encoder.pkl")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


if __name__ == "__main__":
    main()

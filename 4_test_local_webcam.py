"""
4_test_local_webcam.py
======================
Real-time sign language recognition using webcam.

Captures video from local webcam, extracts hand landmarks using MediaPipe Vision Tasks,
runs inference with the trained PyTorch model, and displays predictions in real-time.

Input:  models/sign_model.pt and models/label_encoder.pkl
Output: Live video feed with predictions

Controls:
- Press 'q' to quit
- Press 's' to save a screenshot

Usage:
    python 4_test_local_webcam.py
"""

import os
import cv2

# Keep terminal output clean by hiding noisy non-critical backend warnings.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import logging
from collections import deque
from datetime import datetime
from difflib import get_close_matches
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = Path("models/sign_model.pt")
LABEL_ENCODER_PATH = Path("models/label_encoder.pkl")
SCALER_PATH = Path("models/feature_scaler.pkl")
CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS_DISPLAY_INTERVAL = 30
CONFIDENCE_THRESHOLD = 0.40  # Lowered to see more predictions
MIN_ACCEPT_CONFIDENCE = 0.55
MIN_ACCEPT_MARGIN = 0.12
SMOOTHING_WINDOW = 5  # For prediction smoothing (vote over N frames)
LETTER_COMMIT_FRAMES = 8  # Frames with stable prediction before committing a letter
CHANGE_UNLOCK_FRAMES = 3  # New letter allowed after stable change from previous letter
RELEASE_UNLOCK_FRAMES = 3  # New letter allowed after brief hand release
NO_HAND_SPACE_FRAMES = 18  # Insert a space after a short pause
MAX_SENTENCE_CHARS = 160
BEAM_WIDTH = 6
LETTER_TOP_K = 3
NUM_FEATURES = 63
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lightweight in-code vocabulary for word correction
COMMON_WORDS = {
    "a", "am", "an", "and", "are", "be", "can", "do", "for", "go", "good",
    "have", "hello", "help", "how", "i", "is", "it", "me", "my", "name", "need",
    "no", "not", "please", "she", "sorry", "thank", "thanks", "that", "the", "this",
    "to", "understand", "want", "we", "what", "where", "who", "why", "yes", "you",
    "your"
}

COMMON_BIGRAMS = {
    ("how", "are"), ("are", "you"), ("thank", "you"), ("i", "am"),
    ("my", "name"), ("what", "is"), ("where", "is"), ("please", "help"),
    ("i", "need"), ("can", "you"), ("i", "want"), ("help", "me")
}

# Output configuration
OUTPUT_DIR = Path("outputs")
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.5
FONT_COLOR = (0, 255, 0)  # BGR - Green
FONT_THICKNESS = 2

# Dashboard layout (custom UI)
DASHBOARD_WIDTH = 1600
DASHBOARD_HEIGHT = 900
PANEL_MARGIN = 30
CAM_PANEL_WIDTH = 960
CAM_PANEL_HEIGHT = 540
RESET_BUTTON_WIDTH = 240
RESET_BUTTON_HEIGHT = 52

# Colors (BGR)
BG_COLOR = (20, 20, 24)
PANEL_BG = (36, 36, 44)
PANEL_BORDER = (0, 180, 255)
ACCENT = (0, 220, 255)
SUBTEXT = (180, 180, 190)
BUTTON_BG = (0, 95, 180)
BUTTON_HOVER = (0, 120, 220)
BUTTON_TEXT = (255, 255, 255)


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


class SignLanguageDetector:
    """Real-time sign language detection from video feed."""
    
    def __init__(self, model_path, label_encoder_path, scaler_path):
        """Initialize the detector."""
        # Load model
        logger.info(f"Loading model from {model_path}...")
        self.model = self._load_pytorch_model(model_path)
        self.model.eval()
        logger.info(f"✓ Model loaded (device: {DEVICE})")
        
        # Load label encoder
        logger.info(f"Loading labels from {label_encoder_path}...")
        self.label_encoder = joblib.load(label_encoder_path)
        logger.info(f"Classes: {self.label_encoder.classes_}")

        # Load feature scaler (trained on normalized landmarks)
        logger.info(f"Loading feature scaler from {scaler_path}...")
        self.scaler = joblib.load(scaler_path)
        
        # Initialize MediaPipe Hand Landmarker
        self._init_mediapipe()
        
        # Metrics
        self.frame_count = 0
        self.fps = 0
        self.probability_history = deque(maxlen=SMOOTHING_WINDOW)
        self.no_hand_frames = 0

        # Sentence composition state
        self.composed_text = ""
        self.current_word = ""
        self.word_hypotheses = [("", 0.0)]
        self.stable_label = None
        self.stable_count = 0
        self.last_committed_label = None
        self.ready_for_new_letter = True
        self.change_unlock_count = 0
        self.no_hand_streak = 0

        # Dashboard UI interaction state
        self.reset_button_rect = None
        self.reset_button_hover = False
        self.reset_requested = False

    def reset_statement(self):
        """Clear current statement and decoder state."""
        self.composed_text = ""
        self.current_word = ""
        self.word_hypotheses = [("", 0.0)]
        self.last_committed_label = None
        self.ready_for_new_letter = True
        self.change_unlock_count = 0
        self.stable_label = None
        self.stable_count = 0
        self.no_hand_streak = 0
        self.probability_history.clear()

    def _on_mouse(self, event, x, y, flags, param):
        """Mouse callback for dashboard button interactions."""
        if self.reset_button_rect is None:
            return

        x1, y1, x2, y2 = self.reset_button_rect
        inside = x1 <= x <= x2 and y1 <= y <= y2
        self.reset_button_hover = inside

        if event == cv2.EVENT_LBUTTONDOWN and inside:
            self.reset_requested = True

    @staticmethod
    def _prune_hypotheses(hypotheses, beam_width=BEAM_WIDTH):
        """Keep top scoring unique hypotheses."""
        best_by_text = {}
        for text, score in hypotheses:
            if text not in best_by_text or score > best_by_text[text]:
                best_by_text[text] = score

        items = sorted(best_by_text.items(), key=lambda x: x[1], reverse=True)
        return items[:beam_width] if items else [("", 0.0)]

    def _last_committed_word(self):
        """Get previous committed word from sentence buffer."""
        text = self.composed_text.strip()
        if not text:
            return ""
        parts = text.split()
        return parts[-1].lower() if parts else ""

    def _best_corrected_word_from_beam(self):
        """Select best corrected word from hypotheses using lexical and phrase scoring."""
        prev_word = self._last_committed_word()
        best_word = self.current_word.lower() if self.current_word else ""
        best_score = float("-inf")

        for raw_word, beam_score in self.word_hypotheses:
            if not raw_word:
                continue

            corrected = self.correct_word(raw_word)
            score = float(beam_score)

            # Prefer known words and smoother corrections
            if corrected in COMMON_WORDS:
                score += 0.60
            score -= 0.12 * abs(len(corrected) - len(raw_word))

            # Prefer common phrase transitions
            if prev_word and (prev_word, corrected) in COMMON_BIGRAMS:
                score += 0.45

            if score > best_score:
                best_score = score
                best_word = corrected

        return best_word

    def correct_word(self, word):
        """Apply lightweight dictionary correction to a completed word."""
        if not word:
            return ""

        w = word.lower()
        if len(w) <= 2 or w in COMMON_WORDS:
            return w

        candidates = get_close_matches(w, list(COMMON_WORDS), n=1, cutoff=0.72)
        return candidates[0] if candidates else w

    def get_word_suggestions(self, max_suggestions=3):
        """Return top probable word completions for the current in-progress word."""
        fragment = (self.current_word or "").lower().strip()
        prev_word = self._last_committed_word()
        candidate_scores = {}

        # Score candidates from beam search hypotheses.
        for raw_word, beam_score in self.word_hypotheses:
            if not raw_word:
                continue

            candidate = self.correct_word(raw_word.lower())
            score = float(beam_score)

            if candidate in COMMON_WORDS:
                score += 0.50
            if fragment and candidate.startswith(fragment):
                score += 0.90
            if prev_word and (prev_word, candidate) in COMMON_BIGRAMS:
                score += 0.45

            if candidate not in candidate_scores or score > candidate_scores[candidate]:
                candidate_scores[candidate] = score

        # Add dictionary prefix matches so minor letter errors still get good options.
        if fragment:
            for word in COMMON_WORDS:
                if word.startswith(fragment):
                    score = 0.80
                    if prev_word and (prev_word, word) in COMMON_BIGRAMS:
                        score += 0.35
                    if word not in candidate_scores or score > candidate_scores[word]:
                        candidate_scores[word] = score

            close_words = get_close_matches(fragment, list(COMMON_WORDS), n=6, cutoff=0.60)
            for i, word in enumerate(close_words):
                score = 0.70 - (i * 0.06)
                if prev_word and (prev_word, word) in COMMON_BIGRAMS:
                    score += 0.30
                if word not in candidate_scores or score > candidate_scores[word]:
                    candidate_scores[word] = score

        if not candidate_scores:
            return []

        ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in ranked[:max_suggestions]]

    def finalize_current_word(self):
        """Commit buffered letters as a word with correction and spacing."""
        if not self.current_word and all(not w for w, _ in self.word_hypotheses):
            return

        corrected = self._best_corrected_word_from_beam()

        if self.composed_text and not self.composed_text.endswith(" "):
            self.composed_text += " "

        self.composed_text += corrected
        self.composed_text = self.composed_text[-MAX_SENTENCE_CHARS:]

        # Keep one trailing space between words while composing
        if not self.composed_text.endswith(" "):
            self.composed_text += " "

        self.current_word = ""
        self.word_hypotheses = [("", 0.0)]
        self.last_committed_label = None
        self.ready_for_new_letter = True
        self.change_unlock_count = 0

    def _get_wrapped_text(self, text, max_chars=48, max_lines=2):
        """Wrap text for overlay display."""
        if not text:
            return [""]

        words = text.split(" ")
        lines = []
        current = ""

        for word in words:
            candidate = word if not current else f"{current} {word}"
            if len(candidate) <= max_chars:
                current = candidate
            else:
                lines.append(current)
                current = word
                if len(lines) >= max_lines:
                    break

        if len(lines) < max_lines and current:
            lines.append(current)

        return lines[-max_lines:] if lines else [""]

    def update_sentence(self, smoothed_label, hand_found, mean_probs=None):
        """Convert stable letter predictions into sentence text."""
        if hand_found:
            self.no_hand_streak = 0

            if smoothed_label is None:
                self.stable_label = None
                self.stable_count = 0
                return

            # Unlock next letter only after a stable change from last committed label
            if self.last_committed_label is not None and smoothed_label != self.last_committed_label:
                self.change_unlock_count += 1
                if self.change_unlock_count >= CHANGE_UNLOCK_FRAMES:
                    self.ready_for_new_letter = True
            else:
                self.change_unlock_count = 0

            if smoothed_label == self.stable_label:
                self.stable_count += 1
            else:
                self.stable_label = smoothed_label
                self.stable_count = 1

            ready_to_commit = (
                self.stable_count >= LETTER_COMMIT_FRAMES
                and self.ready_for_new_letter
            )

            if ready_to_commit:
                # Beam update with top-k letters from averaged probabilities
                if mean_probs is not None:
                    top_k = min(LETTER_TOP_K, len(mean_probs))
                    top_indices = np.argpartition(mean_probs, -top_k)[-top_k:]
                    top_indices = top_indices[np.argsort(mean_probs[top_indices])[::-1]]

                    expanded = []
                    for prefix, score in self.word_hypotheses:
                        for idx in top_indices:
                            ch = self.label_encoder.classes_[int(idx)]
                            prob = float(mean_probs[int(idx)])
                            expanded.append((prefix + ch, score + np.log(max(prob, 1e-6))))

                    self.word_hypotheses = self._prune_hypotheses(expanded, BEAM_WIDTH)
                    self.current_word = self.word_hypotheses[0][0][-32:]
                else:
                    self.current_word += smoothed_label
                    self.current_word = self.current_word[-32:]

                self.last_committed_label = smoothed_label
                self.ready_for_new_letter = False
                self.stable_count = 0
                self.change_unlock_count = 0
        else:
            self.no_hand_streak += 1
            self.stable_label = None
            self.stable_count = 0

            if self.no_hand_streak >= RELEASE_UNLOCK_FRAMES:
                self.ready_for_new_letter = True

            if self.no_hand_streak >= NO_HAND_SPACE_FRAMES:
                self.finalize_current_word()
                self.no_hand_streak = 0

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
    
    def _load_pytorch_model(self, model_path):
        """Load PyTorch model from state dict."""
        num_classes = len(joblib.load(LABEL_ENCODER_PATH).classes_)
        model = SignLanguageNet(num_classes=num_classes).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        return model
    
    def _init_mediapipe(self):
        """Initialize MediaPipe Hand Landmarker with Vision Tasks API."""
        from mediapipe.tasks.python import vision
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
        from mediapipe.tasks.python import BaseOptions
        base_options = BaseOptions(model_asset_path=str(model_file))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
    
    def extract_landmarks(self, frame):
        """Extract hand landmarks from frame using MediaPipe Vision Tasks."""
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
                return None, False, detection_result
            
            # Extract landmarks from first hand
            hand_landmarks = detection_result.hand_landmarks[0]
            
            # Flatten to 63-element array
            landmarks = np.array([
                [lm.x, lm.y, lm.z]
                for lm in hand_landmarks
            ]).flatten().astype(np.float32)
            
            return landmarks, True, detection_result
            
        except Exception as e:
            logger.error(f"Error extracting landmarks: {e}")
            return None, False, None
    
    def predict(self, landmarks):
        """Run model inference with PyTorch."""
        try:
            # Apply same preprocessing used during training
            norm_landmarks = self.normalize_landmarks(landmarks)
            scaled_landmarks = self.scaler.transform(norm_landmarks.reshape(1, -1)).astype(np.float32)

            # Convert to tensor
            input_tensor = torch.FloatTensor(scaled_landmarks).to(DEVICE)
            
            # Run inference
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence = probabilities.max().item()
                top2_values, top2_indices = torch.topk(probabilities, k=2, dim=1)
                predicted_idx = top2_indices[0, 0].item()
                margin = (top2_values[0, 0] - top2_values[0, 1]).item()

            # Reject uncertain predictions to reduce bias/misclassification
            if confidence < MIN_ACCEPT_CONFIDENCE or margin < MIN_ACCEPT_MARGIN:
                return None, float(confidence), probabilities.squeeze(0).cpu().numpy()
            
            # Get label
            predicted_label = self.label_encoder.classes_[predicted_idx]
            
            return predicted_label, float(confidence), probabilities.squeeze(0).cpu().numpy()
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None, 0.0, None
    
    def get_smoothed_prediction(self):
        """Get prediction by averaging class probabilities across recent frames."""
        if not self.probability_history:
            return None, 0.0, None

        mean_probs = np.mean(np.stack(self.probability_history, axis=0), axis=0)
        best_idx = int(np.argmax(mean_probs))
        avg_confidence = float(mean_probs[best_idx])

        if avg_confidence < MIN_ACCEPT_CONFIDENCE:
            return None, avg_confidence, mean_probs

        smoothed_label = self.label_encoder.classes_[best_idx]
        return smoothed_label, avg_confidence, mean_probs
    
    def draw_info(self, frame, fps, landmark_status, prediction, confidence):
        """Draw information on frame."""
        frame_copy = frame.copy()
        
        # FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame_copy, fps_text, (10, 40),
                   FONT, 1.0, (255, 255, 255), 2)
        
        # Landmark status
        status_color = (0, 255, 0) if "detected" in landmark_status else (0, 0, 255)
        cv2.putText(frame_copy, landmark_status, (10, 80),
                   FONT, 1.0, status_color, 2)
        
        # Always show prediction with color coding based on confidence
        if prediction is not None:
            pred_text = f"Predicted: {prediction} ({confidence*100:.1f}%)"
            # Green if above threshold, yellow if borderline, red if low
            if confidence > CONFIDENCE_THRESHOLD:
                color = (0, 255, 0)  # Green
            elif confidence > 0.30:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red
            cv2.putText(frame_copy, pred_text, (10, 120),
                       FONT, 1.5, color, FONT_THICKNESS)
            
            # Show threshold indicator
            threshold_text = f"Threshold: {CONFIDENCE_THRESHOLD*100:.0f}%"
            cv2.putText(frame_copy, threshold_text, (10, 150),
                       FONT, 0.8, (200, 200, 200), 1)
        else:
            # No confident prediction (no hand or uncertain class)
            cv2.putText(frame_copy, "No confident prediction", (10, 120),
                       FONT, 1.0, (0, 0, 255), FONT_THICKNESS)
        
        # Frame counter
        frame_text = f"Frame: {self.frame_count}"
        cv2.putText(frame_copy, frame_text, (10, 180),
                   FONT, 0.7, (255, 255, 255), 1)

        # Sentence composition overlay
        display_text = (self.composed_text + self.current_word).strip()
        cv2.putText(frame_copy, "Sentence:", (10, 215), FONT, 0.8, (255, 255, 255), 2)
        sentence_lines = self._get_wrapped_text(display_text)
        y = 245
        for line in sentence_lines:
            cv2.putText(frame_copy, line, (10, y), FONT, 0.9, (0, 255, 255), 2)
            y += 30

        cv2.putText(frame_copy, f"Current word: {self.current_word}", (10, y + 5),
                   FONT, 0.7, (180, 255, 180), 1)

        controls = "Controls: q=quit s=screenshot c=clear Backspace=delete Space=commit-word"
        cv2.putText(frame_copy, controls, (10, frame_copy.shape[0] - 20),
                   FONT, 0.5, (200, 200, 200), 1)
        
        return frame_copy
    
    def draw_landmarks(self, frame, detection_result, offset=(0,0)):
        """Draw hand landmarks on frame."""
        try:
            if not detection_result or not detection_result.hand_landmarks:
                return frame
            
            # Since we extract from the full frame now, use its true dimensions
            h, w = frame.shape[:2]
            ox, oy = offset
            
            for hand_landmarks in detection_result.hand_landmarks:
                # Draw landmarks
                for i, landmark in enumerate(hand_landmarks):
                    x = int(landmark.x * w) + ox
                    y = int(landmark.y * h) + oy
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                
                # Draw connections (basic hand skeleton)
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                ]
                
                for start, end in connections:
                    x1 = int(hand_landmarks[start].x * w) + ox
                    y1 = int(hand_landmarks[start].y * h) + oy
                    x2 = int(hand_landmarks[end].x * w) + ox
                    y2 = int(hand_landmarks[end].y * h) + oy
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            return frame
        except Exception as e:
            logger.error(f"Error drawing landmarks: {e}")
            return frame

    def render_dashboard(self, camera_frame):
        """Render a custom dashboard and place camera feed at top-right."""
        dashboard = np.full((DASHBOARD_HEIGHT, DASHBOARD_WIDTH, 3), BG_COLOR, dtype=np.uint8)

        # Header block
        cv2.putText(dashboard, "PROJECT ONE", (PANEL_MARGIN, 90),
                   FONT, 2.0, ACCENT, 4)
        cv2.putText(dashboard, "E&TC SGSITS", (PANEL_MARGIN, 145),
                   FONT, 1.2, SUBTEXT, 2)

        # Left info panel background
        left_x1 = PANEL_MARGIN
        left_y1 = 190
        left_x2 = DASHBOARD_WIDTH - CAM_PANEL_WIDTH - (PANEL_MARGIN * 2)
        left_y2 = DASHBOARD_HEIGHT - PANEL_MARGIN
        cv2.rectangle(dashboard, (left_x1, left_y1), (left_x2, left_y2), PANEL_BG, -1)
        cv2.rectangle(dashboard, (left_x1, left_y1), (left_x2, left_y2), PANEL_BORDER, 2)

        # Camera panel at top-right
        cam_x1 = DASHBOARD_WIDTH - CAM_PANEL_WIDTH - PANEL_MARGIN
        cam_y1 = PANEL_MARGIN
        cam_x2 = cam_x1 + CAM_PANEL_WIDTH
        cam_y2 = cam_y1 + CAM_PANEL_HEIGHT

        cv2.rectangle(dashboard, (cam_x1 - 3, cam_y1 - 3), (cam_x2 + 3, cam_y2 + 3), PANEL_BORDER, 3)
        cv2.rectangle(dashboard, (cam_x1, cam_y1), (cam_x2, cam_y2), PANEL_BG, -1)

        resized = cv2.resize(camera_frame, (CAM_PANEL_WIDTH, CAM_PANEL_HEIGHT), interpolation=cv2.INTER_LINEAR)
        dashboard[cam_y1:cam_y2, cam_x1:cam_x2] = resized

        # Panel labels
        cv2.putText(dashboard, "LIVE CAMERA", (cam_x1 + 15, cam_y1 - 10),
                   FONT, 0.7, SUBTEXT, 2)
        cv2.putText(dashboard, "PREDICTED SENTENCE", (left_x1 + 15, left_y1 + 35),
                   FONT, 0.8, SUBTEXT, 2)

        # Sentence preview in left panel
        sentence = (self.composed_text + self.current_word).strip()
        lines = self._get_wrapped_text(sentence, max_chars=55, max_lines=10)
        y = left_y1 + 80
        for line in lines:
            cv2.putText(dashboard, line, (left_x1 + 20, y), FONT, 0.8, (0, 255, 255), 2)
            y += 34

        # Show top probable word suggestions to absorb minor letter errors.
        suggestions = self.get_word_suggestions(max_suggestions=3)
        suggestion_title_y = min(y + 18, left_y2 - 90)
        cv2.putText(dashboard, "Probable words:", (left_x1 + 20, suggestion_title_y),
                   FONT, 0.7, SUBTEXT, 2)

        if suggestions:
            suggestion_text = "   ".join([f"{i + 1}.{w}" for i, w in enumerate(suggestions)])
        else:
            suggestion_text = "1.-   2.-   3.-"

        cv2.putText(dashboard, suggestion_text, (left_x1 + 20, suggestion_title_y + 32),
                   FONT, 0.7, (180, 255, 180), 2)

        # Reset button in left panel
        btn_x1 = left_x2 - RESET_BUTTON_WIDTH - 20
        btn_y1 = left_y1 + 15
        btn_x2 = btn_x1 + RESET_BUTTON_WIDTH
        btn_y2 = btn_y1 + RESET_BUTTON_HEIGHT
        self.reset_button_rect = (btn_x1, btn_y1, btn_x2, btn_y2)

        btn_color = BUTTON_HOVER if self.reset_button_hover else BUTTON_BG
        cv2.rectangle(dashboard, (btn_x1, btn_y1), (btn_x2, btn_y2), btn_color, -1)
        cv2.rectangle(dashboard, (btn_x1, btn_y1), (btn_x2, btn_y2), PANEL_BORDER, 2)
        cv2.putText(dashboard, "RESET STATEMENT", (btn_x1 + 22, btn_y1 + 34),
                   FONT, 0.75, BUTTON_TEXT, 2)

        controls = "q: Quit   c: Clear   Backspace: Delete   Space: Commit Word   s: Screenshot"
        cv2.putText(dashboard, controls, (PANEL_MARGIN, DASHBOARD_HEIGHT - 18),
                   FONT, 0.55, SUBTEXT, 1)

        return dashboard
    
    def run(self, camera_id=0):
        """Run real-time detection loop."""
        window_name = "Sign Language Interpreter"
        logger.info(f"Opening camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error(f"Cannot open camera {camera_id}")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info("Starting live detection. Press 'q' quit, 's' screenshot, 'c' clear, Backspace delete, Space commit word.")
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._on_mouse)
        
        import time
        frame_times = deque(maxlen=FPS_DISPLAY_INTERVAL)
        OUTPUT_DIR.mkdir(exist_ok=True)
        
        try:
            while True:
                start_time = time.time()
                ret, frame = cap.read()
                
                if not ret:
                    logger.error("Failed to read frame")
                    break
                
                # Flip frame horizontally to act like a mirror
                frame = cv2.flip(frame, 1)
                
                self.frame_count += 1
                
                # Create a square box on the top right as a visual guide
                box_size = 600
                h, w = frame.shape[:2]
                box_x = max(0, w - box_size - 20)
                box_y = max(0, 20)
                
                # Extract landmarks from the FULL FRAME for max stability. 
                # (Cropping breaks MediaPipe if the hand slightly leaves the box)
                landmarks, hand_found, detection_result = self.extract_landmarks(frame)
                
                # Make prediction
                if hand_found and landmarks is not None:
                    predicted_label, confidence, probs = self.predict(landmarks)
                    if probs is not None:
                        self.probability_history.append(probs)
                    self.no_hand_frames = 0
                    landmark_status = "✓ Hand detected"
                else:
                    self.no_hand_frames += 1
                    landmark_status = "✗ No hand detected"
                    # Clear prediction history if no hand for several frames
                    if self.no_hand_frames > 2:
                        self.probability_history.clear()
                
                # Get smoothed prediction
                smoothed_label, smoothed_conf, mean_probs = self.get_smoothed_prediction()

                # Update sentence builder from smoothed predictions
                self.update_sentence(smoothed_label, hand_found, mean_probs)
                
                # Calculate FPS
                frame_times.append(time.time() - start_time)
                if len(frame_times) == FPS_DISPLAY_INTERVAL:
                    self.fps = FPS_DISPLAY_INTERVAL / sum(frame_times)
                
                # Draw information
                display_frame = self.draw_info(
                    frame, self.fps, landmark_status, smoothed_label, smoothed_conf
                )
                
                # Draw the static target box to guide the user
                cv2.rectangle(display_frame, (box_x, box_y), (box_x + box_size, box_y + box_size), (0, 200, 255), 3)
                cv2.putText(display_frame, "Place Hand Inside for Presentation", (box_x + 10, box_y + 30), FONT, 0.7, (0, 200, 255), 2)
                
                # Draw landmarks on the full frame
                if detection_result:
                    display_frame = self.draw_landmarks(display_frame, detection_result, offset=(0, 0))

                # Render themed dashboard with camera in top-right panel
                dashboard_frame = self.render_dashboard(display_frame)
                
                # Display frame
                cv2.imshow(window_name, dashboard_frame)

                # Handle mouse-triggered reset
                if self.reset_requested:
                    self.reset_statement()
                    self.reset_requested = False
                    logger.info("Composed text cleared (button)")
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit signal received")
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = OUTPUT_DIR / f"screenshot_{timestamp}.png"
                    cv2.imwrite(str(filename), dashboard_frame)
                    logger.info(f"Screenshot saved: {filename}")
                elif key == ord('c'):
                    self.reset_statement()
                    logger.info("Composed text cleared")
                elif key in (8, 127):  # Backspace on different keyboards
                    if self.current_word:
                        self.current_word = self.current_word[:-1]
                        trimmed = []
                        for text, score in self.word_hypotheses:
                            trimmed.append((text[:-1] if text else "", score))
                        self.word_hypotheses = self._prune_hypotheses(trimmed, BEAM_WIDTH)
                    else:
                        # Remove from sentence tail if no active word
                        self.composed_text = self.composed_text.rstrip()
                        self.composed_text = self.composed_text[:-1]
                    logger.info(f"Composed text: {(self.composed_text + self.current_word).strip()}")
                elif key == ord(' '):
                    self.finalize_current_word()
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Webcam closed")
            logger.info(f"Total frames processed: {self.frame_count}")
            if self.frame_count > 0:
                logger.info(f"Frames without hand: {self.no_hand_frames} ({self.no_hand_frames/self.frame_count*100:.1f}%)")
            if self.composed_text.strip():
                logger.info(f"Composed sentence: {(self.composed_text + self.current_word).strip()}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Sign Language Interpreter - Local Testing")
    print("=" * 60)
    
    try:
        if not MODEL_PATH.exists():
            logger.error(f"Model not found: {MODEL_PATH}")
            logger.error("Please run 3_train_model.py first")
            return False
        
        if not LABEL_ENCODER_PATH.exists():
            logger.error(f"Label encoder not found: {LABEL_ENCODER_PATH}")
            logger.error("Please run 3_train_model.py first")
            return False

        if not SCALER_PATH.exists():
            logger.error(f"Feature scaler not found: {SCALER_PATH}")
            logger.error("Please run 3_train_model.py first")
            return False
        
        detector = SignLanguageDetector(MODEL_PATH, LABEL_ENCODER_PATH, SCALER_PATH)
        detector.run(CAMERA_ID)
        
        print("\n" + "=" * 60)
        print("✓ Testing completed!")
        print(f"  Next step: Run 5_convert_tflite.py for edge deployment")
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

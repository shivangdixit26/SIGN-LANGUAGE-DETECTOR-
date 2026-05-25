"""
2_extract_features.py
=====================
Extracts hand landmarks from image dataset using MediaPipe.

Processes all images in the dataset directory, extracts 21 hand landmarks
per image using MediaPipe, and saves features to keypoints.csv.

Features per image: 21 landmarks × 3 coordinates (x, y, z) = 63 values

Input:  data/asl_alphabet/ (image directory structure)
Output: keypoints.csv (features and labels)

Usage:
    python 2_extract_features.py
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
import mediapipe as mp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path("data/asl_alphabet")
OUTPUT_FILE = Path("keypoints.csv")
MAX_HANDS = 1  # Detect only one hand per image
HAND_CONFIDENCE_THRESHOLD = 0.5  # Min confidence for hand detection


class HandLandmarkExtractor:
    """Extracts hand landmarks from images using MediaPipe 0.10+ Vision API."""
    
    def __init__(self):
        """Initialize MediaPipe Hand Landmarker."""
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        import urllib.request
        
        # Get or download hand landmarker model
        model_path = self._get_model_path()
        
        # Create hand landmarker with model path
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=HAND_CONFIDENCE_THRESHOLD,
            min_hand_presence_confidence=HAND_CONFIDENCE_THRESHOLD,
            min_tracking_confidence=HAND_CONFIDENCE_THRESHOLD
        )
        
        self.landmarker = vision.HandLandmarker.create_from_options(options)
    
    def _get_model_path(self):
        """Get or download the hand landmarker model."""
        from pathlib import Path
        import urllib.request
        import os
        
        # Model cache directory
        cache_dir = Path.home() / ".cache" / "mediapipe" / "vision"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        model_file = cache_dir / "hand_landmarker.task"
        
        # If already cached, return it
        if model_file.exists():
            return str(model_file)
        
        # Download model from Google's MediaPipe repository
        model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        
        logger.info(f"Downloading hand landmarker model ({model_url})")
        try:
            urllib.request.urlretrieve(model_url, model_file)
            logger.info(f"✓ Model downloaded: {model_file}")
            return str(model_file)
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            logger.error("You can manually download from:")
            logger.error(model_url)
            raise
    
    def extract_landmarks(self, image_path):
        """
        Extract hand landmarks from an image.
        
        Args:
            image_path (Path): Path to the image file
            
        Returns:
            tuple: (landmarks_array, success_flag)
                   landmarks_array: 63-element array (21 landmarks × 3 coords)
                   success_flag: True if hand detected, False otherwise
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Failed to read image: {image_path}")
                return None, False
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image using the correct API
            import mediapipe as mp
            from mediapipe.tasks.python.vision import HandLandmarker
            
            # MediaPipe expects uint8 numpy array
            if image_rgb.dtype != np.uint8:
                image_rgb = (image_rgb * 255).astype(np.uint8)
            
            # Create image with proper format - use PIL/numpy approach
            from mediapipe.tasks.python.vision.core.image_processing_options import ImageProcessingOptions
            
            # The landmarker.detect() expects a cv2 image directly in recent versions
            # But we need to wrap it properly
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=image_rgb
            )
            
            # Process image with MediaPipe
            detection_result = self.landmarker.detect(mp_image)
            
            # Check if hand was detected
            if not detection_result.hand_landmarks:
                return None, False
            
            # Extract landmarks from the first detected hand
            hand_landmarks = detection_result.hand_landmarks[0]
            
            # Flatten landmarks into 63-element array (21 points × 3 coords: x, y, z)
            landmarks_array = np.array([
                [lm.x, lm.y, lm.z]
                for lm in hand_landmarks
            ]).flatten()
            
            return landmarks_array, True
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return None, False
    def close(self):
        """Clean up resources."""
        # MediaPipe 0.10+ doesn't require explicit cleanup
        pass


def find_image_files(directory):
    """
    Find all image files in directory structure.
    
    Args:
        directory (Path): Root directory to search
        
    Returns:
        dict: {label: [list of image paths]}
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    label_images = {}
    
    if not directory.exists():
        logger.error(f"Dataset directory not found: {directory}")
        return {}
    
    # Check if there's an 'asl_dataset' subdirectory
    if (directory / "asl_dataset").exists():
        directory = directory / "asl_dataset"
        logger.info(f"Found asl_dataset subdirectory, using: {directory}")
    
    # Recursively find all images
    for image_path in sorted(directory.rglob("*")):
        if image_path.suffix.lower() in image_extensions:
            # Extract label from directory structure
            # Typically: asl_alphabet/A/... or asl_alphabet/train/A/...
            # Or: asl_alphabet/asl_dataset/a/...
            relative_parts = image_path.relative_to(directory).parts
            
            # Label is usually the first directory
            if relative_parts:
                label = relative_parts[0]
                
                # Handle both letter labels (A-Z, a-z) and numeric labels (0-9)
                if (len(label) == 1 and label.isalpha()) or label.isdigit():
                    # Convert to uppercase for consistency
                    if label.isalpha():
                        label = label.upper()
                    
                    if label not in label_images:
                        label_images[label] = []
                    label_images[label].append(image_path)
    
    return label_images


def extract_all_features(extractor, label_images):
    """
    Extract features from all images.
    
    Args:
        extractor (HandLandmarkExtractor): The extractor instance
        label_images (dict): {label: [image paths]}
        
    Returns:
        pd.DataFrame: DataFrame with features and labels
    """
    features_list = []
    labels_list = []
    skipped_count = 0
    processed_count = 0
    
    # Calculate total images
    total_images = sum(len(paths) for paths in label_images.values())
    
    with tqdm(total=total_images, desc="Extracting landmarks") as pbar:
        for label in sorted(label_images.keys()):
            for image_path in label_images[label]:
                landmarks, success = extractor.extract_landmarks(image_path)
                
                if success:
                    features_list.append(landmarks)
                    labels_list.append(label)
                    processed_count += 1
                else:
                    skipped_count += 1
                
                pbar.update(1)
    
    # Create DataFrame
    df = pd.DataFrame(features_list)
    df.insert(0, 'label', labels_list)
    
    logger.info(f"\nExtraction Summary:")
    logger.info(f"  Successfully processed: {processed_count}")
    logger.info(f"  Skipped (no hand detected): {skipped_count}")
    logger.info(f"  Total samples: {len(df)}")
    logger.info(f"  Labels: {sorted(df['label'].unique())}")
    
    return df


def save_features(df, output_path):
    """
    Save features to CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        output_path (Path): Output CSV path
    """
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"✓ Features saved to: {output_path}")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        return True
    except Exception as e:
        logger.error(f"Error saving features: {e}")
        return False


def main():
    """Main execution flow."""
    print("=" * 60)
    print("Hand Landmark Feature Extraction")
    print("=" * 60)
    
    try:
        # Check if dataset exists
        if not DATA_DIR.exists():
            logger.error(f"\n✗ Dataset directory not found: {DATA_DIR}")
            logger.error("  Please run 1_fetch_data.py first")
            return False
        
        # Initialize extractor
        logger.info("Initializing MediaPipe Hands...")
        extractor = HandLandmarkExtractor()
        
        # Find images
        logger.info(f"Searching for images in {DATA_DIR}...")
        label_images = find_image_files(DATA_DIR)
        
        if not label_images:
            logger.error("✗ No images found in dataset directory")
            return False
        
        logger.info(f"Found {len(label_images)} labels with images")
        for label, paths in sorted(label_images.items()):
            logger.info(f"  {label}: {len(paths)} images")
        
        # Extract features
        logger.info("\n⏳ Extracting hand landmarks from images...")
        df = extract_all_features(extractor, label_images)
        
        # Clean up
        extractor.close()
        
        if len(df) == 0:
            logger.error("\n✗ No features extracted. Check dataset and hand detection.")
            return False
        
        # Save features
        logger.info("\n⏳ Saving features to CSV...")
        if not save_features(df, OUTPUT_FILE):
            return False
        
        print("\n" + "=" * 60)
        print("✓ Feature extraction completed successfully!")
        print(f"  Output: {OUTPUT_FILE}")
        print(f"  Samples: {len(df)}")
        print(f"  Feature dimension: 63 (21 landmarks × 3 coords)")
        print(f"  Next step: Run 3_train_model.py")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"\n✗ Error during feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

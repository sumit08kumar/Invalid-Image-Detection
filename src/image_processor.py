"""
Invalid Image Detection System - Main Image Processing Module
Implements blur detection, duplicate detection, and fake/screenshot classification
"""

import cv2
import numpy as np
import os
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import imagehash
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BlurDetector:
    """Detects blurry images using Laplacian variance method"""
    
    def __init__(self, threshold=100.0):
        self.threshold = threshold
    
    def detect_blur(self, image_path):
        """
        Detect if an image is blurry using Laplacian variance
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Contains is_blurry (bool) and variance (float)
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return {"is_blurry": True, "variance": 0.0, "error": "Could not read image"}
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Determine if blurry
            is_blurry = bool(laplacian_var < self.threshold)
            
            return {
                "is_blurry": is_blurry,
                "variance": float(laplacian_var),
                "threshold": float(self.threshold)
            }
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return {"is_blurry": True, "variance": 0.0, "error": str(e)}

class DuplicateDetector:
    """Detects duplicate and near-duplicate images using SSIM and perceptual hashing"""
    
    def __init__(self, ssim_threshold=0.95, hash_threshold=5):
        self.ssim_threshold = ssim_threshold
        self.hash_threshold = hash_threshold
        self.image_hashes = {}
    
    def calculate_ssim(self, image1_path, image2_path):
        """
        Calculate SSIM between two images
        
        Args:
            image1_path (str): Path to first image
            image2_path (str): Path to second image
            
        Returns:
            float: SSIM score between 0 and 1
        """
        try:
            # Read images
            img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
            
            if img1 is None or img2 is None:
                return 0.0
            
            # Resize images to same size for comparison
            height = min(img1.shape[0], img2.shape[0])
            width = min(img1.shape[1], img2.shape[1])
            
            img1_resized = cv2.resize(img1, (width, height))
            img2_resized = cv2.resize(img2, (width, height))
            
            # Calculate SSIM
            ssim_score = ssim(img1_resized, img2_resized)
            return float(ssim_score)
            
        except Exception as e:
            logger.error(f"Error calculating SSIM: {str(e)}")
            return 0.0
    
    def calculate_perceptual_hash(self, image_path):
        """
        Calculate perceptual hash for an image
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            str: Perceptual hash as string
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Calculate perceptual hash
                phash = imagehash.phash(img)
                return str(phash)
                
        except Exception as e:
            logger.error(f"Error calculating hash for {image_path}: {str(e)}")
            return None
    
    def find_duplicates(self, image_paths):
        """
        Find duplicate images in a list of image paths
        
        Args:
            image_paths (list): List of image file paths
            
        Returns:
            dict: Dictionary containing duplicate groups and statistics
        """
        duplicates = []
        processed_hashes = {}
        
        for i, image_path in enumerate(image_paths):
            # Calculate hash for current image
            current_hash = self.calculate_perceptual_hash(image_path)
            if current_hash is None:
                continue
            
            # Check for duplicates
            is_duplicate = False
            for existing_hash, existing_paths in processed_hashes.items():
                # Calculate Hamming distance between hashes
                hash_distance = bin(int(current_hash, 16) ^ int(existing_hash, 16)).count('1')
                
                if hash_distance <= self.hash_threshold:
                    # Found a duplicate
                    existing_paths.append(image_path)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                processed_hashes[current_hash] = [image_path]
        
        # Extract duplicate groups (groups with more than 1 image)
        duplicate_groups = []
        for hash_val, paths in processed_hashes.items():
            if len(paths) > 1:
                duplicate_groups.append({
                    "hash": hash_val,
                    "images": paths,
                    "count": len(paths)
                })
        
        return {
            "duplicate_groups": duplicate_groups,
            "total_duplicates": sum(group["count"] - 1 for group in duplicate_groups),
            "unique_images": len(processed_hashes)
        }

class FakeScreenshotDetector:
    """Detects fake images and screenshots using a CNN model"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.input_shape = (224, 224, 3)
        self.model_path = model_path
        
        # Initialize or load model
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.create_model()
    
    def create_model(self):
        """Create a CNN model for fake/screenshot detection using MobileNetV2"""
        try:
            # Load pre-trained MobileNetV2
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
            
            # Freeze base model layers
            base_model.trainable = False
            
            # Add custom classification layers
            model = keras.Sequential([
                base_model,
                keras.layers.GlobalAveragePooling2D(),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(2, activation='softmax')  # 2 classes: real, fake/screenshot
            ])
            
            # Compile model
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            logger.info("Created new CNN model for fake/screenshot detection")
            
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            self.model = None
    
    def load_model(self, model_path):
        """Load a pre-trained model"""
        try:
            self.model = keras.models.load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.create_model()
    
    def save_model(self, model_path):
        """Save the current model"""
        try:
            if self.model:
                self.model.save(model_path)
                logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for model prediction
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            numpy.ndarray: Preprocessed image array
        """
        try:
            # Load and resize image
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
            
            # Convert to array and preprocess
            image_array = img_to_array(image)
            image_array = np.expand_dims(image_array, axis=0)
            image_array = preprocess_input(image_array)
            
            return image_array
            
        except Exception as e:
            logger.error(f"Error preprocessing {image_path}: {str(e)}")
            return None
    
    def predict(self, image_path):
        """
        Predict if an image is fake/screenshot
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            dict: Prediction results with confidence scores
        """
        if self.model is None:
            return {"is_fake": False, "confidence": 0.0, "error": "Model not available"}
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            if processed_image is None:
                return {"is_fake": True, "confidence": 0.0, "error": "Could not process image"}
            
            # Make prediction
            prediction = self.model.predict(processed_image, verbose=0)
            
            # Extract probabilities
            real_prob = float(prediction[0][0])
            fake_prob = float(prediction[0][1])
            
            # Determine classification
            is_fake = bool(fake_prob > real_prob)
            confidence = float(max(real_prob, fake_prob))
            
            return {
                "is_fake": is_fake,
                "confidence": confidence,
                "real_probability": real_prob,
                "fake_probability": fake_prob
            }
            
        except Exception as e:
            logger.error(f"Error predicting {image_path}: {str(e)}")
            return {"is_fake": False, "confidence": 0.0, "error": str(e)}

class ImageProcessor:
    """Main class that orchestrates all image processing tasks"""
    
    def __init__(self, blur_threshold=100.0, ssim_threshold=0.95, hash_threshold=5):
        self.blur_detector = BlurDetector(blur_threshold)
        self.duplicate_detector = DuplicateDetector(ssim_threshold, hash_threshold)
        self.fake_detector = FakeScreenshotDetector()
        
    def process_single_image(self, image_path):
        """
        Process a single image for all detection tasks
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            dict: Complete analysis results
        """
        results = {
            "image_path": image_path,
            "filename": os.path.basename(image_path),
            "timestamp": datetime.now().isoformat()
        }
        
        # Blur detection
        blur_result = self.blur_detector.detect_blur(image_path)
        results["blur_detection"] = blur_result
        
        # Fake/screenshot detection
        fake_result = self.fake_detector.predict(image_path)
        results["fake_detection"] = fake_result
        
        # Overall validity
        results["is_valid"] = bool(not (
            blur_result.get("is_blurry", True) or 
            fake_result.get("is_fake", False)
        ))
        
        return results
    
    def process_directory(self, directory_path, output_dir="results"):
        """
        Process all images in a directory
        
        Args:
            directory_path (str): Path to directory containing images
            output_dir (str): Directory to save results
            
        Returns:
            dict: Complete processing results and statistics
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_paths = []
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
        
        logger.info(f"Found {len(image_paths)} images to process")
        
        # Process individual images
        individual_results = []
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            result = self.process_single_image(image_path)
            individual_results.append(result)
        
        # Find duplicates
        logger.info("Detecting duplicates...")
        duplicate_results = self.duplicate_detector.find_duplicates(image_paths)
        
        # Generate statistics
        stats = self._generate_statistics(individual_results, duplicate_results)
        
        # Compile final results
        final_results = {
            "processing_info": {
                "total_images": len(image_paths),
                "processed_at": datetime.now().isoformat(),
                "input_directory": directory_path,
                "output_directory": output_dir
            },
            "individual_results": individual_results,
            "duplicate_detection": duplicate_results,
            "statistics": stats
        }
        
        # Save results
        self._save_results(final_results, output_dir)
        
        return final_results
    
    def _generate_statistics(self, individual_results, duplicate_results):
        """Generate processing statistics"""
        total_images = len(individual_results)
        
        # Count issues
        blurry_count = sum(1 for r in individual_results if r["blur_detection"].get("is_blurry", False))
        fake_count = sum(1 for r in individual_results if r["fake_detection"].get("is_fake", False))
        valid_count = sum(1 for r in individual_results if r.get("is_valid", False))
        duplicate_count = duplicate_results.get("total_duplicates", 0)
        
        # Calculate percentages
        stats = {
            "total_images": total_images,
            "valid_images": valid_count,
            "blurry_images": blurry_count,
            "fake_images": fake_count,
            "duplicate_images": duplicate_count,
            "unique_images": duplicate_results.get("unique_images", total_images),
            "percentages": {
                "valid": round((valid_count / total_images) * 100, 2) if total_images > 0 else 0,
                "blurry": round((blurry_count / total_images) * 100, 2) if total_images > 0 else 0,
                "fake": round((fake_count / total_images) * 100, 2) if total_images > 0 else 0,
                "duplicate": round((duplicate_count / total_images) * 100, 2) if total_images > 0 else 0
            }
        }
        
        return stats
    
    def _save_results(self, results, output_dir):
        """Save results to various formats"""
        # Save JSON report
        json_path = os.path.join(output_dir, "image_analysis_report.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSV summary
        csv_path = os.path.join(output_dir, "image_analysis_summary.csv")
        df_data = []
        
        for result in results["individual_results"]:
            row = {
                "filename": result["filename"],
                "image_path": result["image_path"],
                "is_valid": result["is_valid"],
                "is_blurry": result["blur_detection"].get("is_blurry", False),
                "blur_variance": result["blur_detection"].get("variance", 0),
                "is_fake": result["fake_detection"].get("is_fake", False),
                "fake_confidence": result["fake_detection"].get("confidence", 0),
                "real_probability": result["fake_detection"].get("real_probability", 0),
                "fake_probability": result["fake_detection"].get("fake_probability", 0)
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(csv_path, index=False)
        
        # Save statistics summary
        stats_path = os.path.join(output_dir, "statistics_summary.txt")
        with open(stats_path, 'w') as f:
            stats = results["statistics"]
            f.write("Image Analysis Statistics\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Total Images Processed: {stats['total_images']}\n")
            f.write(f"Valid Images: {stats['valid_images']} ({stats['percentages']['valid']}%)\n")
            f.write(f"Blurry Images: {stats['blurry_images']} ({stats['percentages']['blurry']}%)\n")
            f.write(f"Fake/Screenshot Images: {stats['fake_images']} ({stats['percentages']['fake']}%)\n")
            f.write(f"Duplicate Images: {stats['duplicate_images']} ({stats['percentages']['duplicate']}%)\n")
            f.write(f"Unique Images: {stats['unique_images']}\n")
        
        logger.info(f"Results saved to {output_dir}")
        logger.info(f"JSON report: {json_path}")
        logger.info(f"CSV summary: {csv_path}")
        logger.info(f"Statistics: {stats_path}")


# Enhanced preprocessing with debugging capabilities

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from skimage.transform import resize
import cv2
import os
import time

class EnhancedPreprocessor:
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        self.debug_folder = "preprocessing_debug"
        
        if debug_mode and not os.path.exists(self.debug_folder):
            os.makedirs(self.debug_folder)
    
    def preprocess_image(self, image, save_debug=False):
        """Enhanced preprocessing with debug visualization"""
        
        # Step 1: Original image
        if self.debug_mode and save_debug:
            self.save_debug_image(image, "1_original.png")
        
        # Step 2: Resize to target size first
        img_size = (128, 128, 3)
        img_resized = resize(np.array(image), img_size)
        
        if self.debug_mode and save_debug:
            self.save_debug_image(img_resized, "2_resized_color.png")
        
        # Step 3: Convert to grayscale
        img_gray = np.dot(img_resized[...,:3], [0.299, 0.587, 0.114])
        
        if self.debug_mode and save_debug:
            self.save_debug_image(img_gray, "3_grayscale.png", cmap='gray')
        
        # Step 4: Final resize (redundant in original code)
        img_gray = resize(img_gray, (128, 128))
        
        if self.debug_mode and save_debug:
            self.save_debug_image(img_gray, "4_final.png", cmap='gray')
        
        return np.expand_dims(img_gray, axis=0)
    
    def preprocess_image_enhanced(self, image, save_debug=False):
        """Better preprocessing that preserves more game details"""
        
        # Step 1: Original image
        if self.debug_mode and save_debug:
            self.save_debug_image(image, "enhanced_1_original.png")
        
        # Step 2: Crop to game area only (remove UI elements if needed)
        # You might want to crop out score/UI elements here
        img_cropped = np.array(image)
        
        # Step 3: Resize to larger size to preserve details
        img_size = (256, 256, 3)  # Larger size preserves more detail
        img_resized = resize(img_cropped, img_size)
        
        if self.debug_mode and save_debug:
            self.save_debug_image(img_resized, "enhanced_2_resized.png")
        
        # Step 4: Enhanced grayscale conversion with contrast enhancement
        img_gray = np.dot(img_resized[...,:3], [0.299, 0.587, 0.114])
        
        # Step 5: Enhance contrast to make obstacles more visible
        img_gray = self.enhance_contrast(img_gray)
        
        if self.debug_mode and save_debug:
            self.save_debug_image(img_gray, "enhanced_3_contrast.png", cmap='gray')
        
        # Step 6: Final resize if needed
        img_gray = resize(img_gray, (256, 256))
        
        return np.expand_dims(img_gray, axis=0)
    
    def enhance_contrast(self, image):
        """Enhance contrast to make game elements more visible"""
        # Normalize to 0-255 range
        img_normalized = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        img_enhanced = clahe.apply(img_normalized)
        
        # Convert back to 0-1 range
        return img_enhanced.astype(np.float32) / 255.0
    
    def save_debug_image(self, image, filename, cmap=None):
        """Save debug image to disk"""
        filepath = os.path.join(self.debug_folder, f"{int(time.time())}_{filename}")
        
        plt.figure(figsize=(8, 8))
        if cmap:
            plt.imshow(image, cmap=cmap)
        else:
            plt.imshow(image)
        plt.axis('off')
        plt.title(filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Debug image saved: {filepath}")
    
    def compare_preprocessing_methods(self, image):
        """Compare original vs enhanced preprocessing"""
        print("Comparing preprocessing methods...")
        
        # Original method
        original_result = self.preprocess_image(image, save_debug=True)
        
        # Enhanced method  
        enhanced_result = self.preprocess_image_enhanced(image, save_debug=True)
        
        # Side by side comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Original preprocessing
        axes[1].imshow(original_result[0], cmap='gray')
        axes[1].set_title('Original Preprocessing\n(128x128)')
        axes[1].axis('off')
        
        # Enhanced preprocessing
        axes[2].imshow(enhanced_result[0], cmap='gray')
        axes[2].set_title('Enhanced Preprocessing\n(256x256 + Contrast)')
        axes[2].axis('off')
        
        plt.tight_layout()
        comparison_path = os.path.join(self.debug_folder, f"{int(time.time())}_comparison.png")
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Comparison saved: {comparison_path}")
        
        return original_result, enhanced_result

# Legacy function for backward compatibility
def preprocess_image(img):
    """Original preprocessing function"""
    img_size = (128,128,3)
    img = resize(np.array(img), img_size)
    img_gray = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    img_gray = resize(img_gray, (128, 128))
    return np.expand_dims(img_gray, axis=0)

# Debug function to test your current preprocessing
def debug_current_preprocessing():
    """Debug the current preprocessing pipeline"""
    print("=== Preprocessing Debug ===")
    
    # Create enhanced preprocessor with debug mode
    preprocessor = EnhancedPreprocessor(debug_mode=True)
    
    # Test with a sample image (you can replace with actual game screenshot)
    try:
        # Try to load a test image
        if os.path.exists('images/temp1.png'):
            test_image = img.imread('images/temp1.png')
            print("Using test image: images/temp1.png")
        else:
            print("No test image found. Creating synthetic test image...")
            # Create a synthetic game-like image for testing
            test_image = np.random.rand(800, 600, 3)
            
        print(f"Original image shape: {test_image.shape}")
        
        # Compare preprocessing methods
        original, enhanced = preprocessor.compare_preprocessing_methods(test_image)
        
        print(f"Original preprocessing output shape: {original.shape}")
        print(f"Enhanced preprocessing output shape: {enhanced.shape}")
        
        print("\nCheck the 'preprocessing_debug' folder for detailed comparison images!")
        
    except Exception as e:
        print(f"Error during preprocessing debug: {e}")

if __name__ == "__main__":
    debug_current_preprocessing()
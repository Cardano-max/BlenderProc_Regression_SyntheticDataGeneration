"""
Post-Processing Script for BlenderProc Synthetic Data
======================================================
Applies post-processing effects to soften perfect CG boundaries
and add realistic camera noise.

Based on Professor's feedback:
- "The boundary is too crisp... adding noise could be an option"
- "It's never on real pictures that crisp"

Author: Maidah Binte Tariq
Date: January 2025
"""

import os
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from tqdm import tqdm

try:
    import cv2
except ImportError:
    print("OpenCV not found. Install with: pip install opencv-python")
    exit(1)

try:
    from PIL import Image, ImageFilter, ImageEnhance
except ImportError:
    print("Pillow not found. Install with: pip install Pillow")
    exit(1)


# ============================================================================
# NOISE FUNCTIONS
# ============================================================================

def add_gaussian_noise(image: np.ndarray, strength: float = 0.02) -> np.ndarray:
    """
    Add Gaussian noise to image.
    
    Args:
        image: Input image (0-255)
        strength: Noise strength (0-1), default 0.02 for subtle noise
    
    Returns:
        Noisy image
    """
    # Generate noise
    noise = np.random.normal(0, strength * 255, image.shape).astype(np.float32)
    
    # Add noise to image
    noisy = image.astype(np.float32) + noise
    
    # Clip to valid range
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    return noisy


def add_salt_pepper_noise(image: np.ndarray, amount: float = 0.002) -> np.ndarray:
    """
    Add salt and pepper noise (simulates camera sensor noise).
    
    Args:
        image: Input image
        amount: Proportion of pixels affected
    
    Returns:
        Noisy image
    """
    noisy = image.copy()
    
    # Salt (white pixels)
    num_salt = int(amount * image.size / 2)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 255
    
    # Pepper (black pixels)
    num_pepper = int(amount * image.size / 2)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 0
    
    return noisy


def add_edge_noise(image: np.ndarray, mask: Optional[np.ndarray] = None, 
                   strength: float = 0.05) -> np.ndarray:
    """
    Add noise specifically to edges/boundaries of objects.
    This addresses the professor's concern about "too crisp" boundaries.
    
    Args:
        image: Input image
        mask: Optional segmentation mask (if available)
        strength: Edge noise strength
    
    Returns:
        Image with noisy edges
    """
    # Detect edges using Canny
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges to create edge region
    kernel = np.ones((5, 5), np.uint8)
    edge_region = cv2.dilate(edges, kernel, iterations=2)
    
    # Create noise for edge region
    noise = np.random.normal(0, strength * 255, image.shape).astype(np.float32)
    
    # Apply noise only to edge regions
    edge_mask = (edge_region > 0).astype(np.float32)
    if len(image.shape) == 3:
        edge_mask = np.stack([edge_mask] * 3, axis=-1)
    
    noisy = image.astype(np.float32) + noise * edge_mask
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    return noisy


def add_chromatic_aberration(image: np.ndarray, shift: int = 2) -> np.ndarray:
    """
    Add subtle chromatic aberration (color fringing) for realism.
    Common in real camera lenses.
    
    Args:
        image: Input BGR image
        shift: Pixel shift amount
    
    Returns:
        Image with chromatic aberration
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        return image
    
    b, g, r = cv2.split(image)
    
    # Shift red and blue channels slightly
    h, w = r.shape
    
    # Shift red channel
    r_shifted = np.zeros_like(r)
    r_shifted[:, shift:] = r[:, :-shift]
    
    # Shift blue channel opposite direction
    b_shifted = np.zeros_like(b)
    b_shifted[:, :-shift] = b[:, shift:]
    
    # Merge with slight blending at edges
    result = cv2.merge([b_shifted, g, r_shifted])
    
    # Blend with original to reduce effect
    result = cv2.addWeighted(image, 0.7, result, 0.3, 0)
    
    return result


# ============================================================================
# COLOR ADJUSTMENT FUNCTIONS
# ============================================================================

def adjust_brightness_contrast(image: np.ndarray, 
                                brightness: float = 0.0,
                                contrast: float = 1.0) -> np.ndarray:
    """
    Adjust image brightness and contrast.
    
    Args:
        image: Input image
        brightness: Brightness adjustment (-1 to 1)
        contrast: Contrast multiplier (0.5 to 1.5 typical)
    
    Returns:
        Adjusted image
    """
    # Convert to float
    img_float = image.astype(np.float32)
    
    # Apply contrast
    img_float = (img_float - 127.5) * contrast + 127.5
    
    # Apply brightness
    img_float = img_float + brightness * 255
    
    # Clip and convert back
    return np.clip(img_float, 0, 255).astype(np.uint8)


def adjust_saturation(image: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """
    Adjust color saturation.
    
    Args:
        image: Input BGR image
        factor: Saturation multiplier (0 = grayscale, 1 = original, >1 = more saturated)
    
    Returns:
        Adjusted image
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Adjust saturation
    hsv[:, :, 1] = hsv[:, :, 1] * factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    
    # Convert back
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_color_jitter(image: np.ndarray, 
                       brightness_range: Tuple[float, float] = (-0.1, 0.1),
                       contrast_range: Tuple[float, float] = (0.9, 1.1),
                       saturation_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    """
    Apply random color jitter for domain randomization.
    
    Args:
        image: Input image
        brightness_range: Random brightness adjustment range
        contrast_range: Random contrast adjustment range  
        saturation_range: Random saturation adjustment range
    
    Returns:
        Color-jittered image
    """
    brightness = np.random.uniform(*brightness_range)
    contrast = np.random.uniform(*contrast_range)
    saturation = np.random.uniform(*saturation_range)
    
    result = adjust_brightness_contrast(image, brightness, contrast)
    result = adjust_saturation(result, saturation)
    
    return result


# ============================================================================
# BLUR FUNCTIONS
# ============================================================================

def add_motion_blur(image: np.ndarray, kernel_size: int = 5, 
                    angle: float = 0) -> np.ndarray:
    """
    Add motion blur effect (simulates camera shake).
    
    Args:
        image: Input image
        kernel_size: Size of motion blur kernel
        angle: Angle of motion in degrees
    
    Returns:
        Motion-blurred image
    """
    # Create motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    
    # Rotate kernel by angle
    center = (kernel_size // 2, kernel_size // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    
    # Apply blur
    return cv2.filter2D(image, -1, kernel)


def add_lens_blur(image: np.ndarray, radius: int = 3) -> np.ndarray:
    """
    Add subtle lens blur (simulates slight focus issues).
    
    Args:
        image: Input image
        radius: Blur radius
    
    Returns:
        Blurred image
    """
    return cv2.GaussianBlur(image, (radius * 2 + 1, radius * 2 + 1), 0)


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_image(image_path: str, output_path: str, config: dict) -> None:
    """
    Apply all post-processing effects to a single image.
    
    Args:
        image_path: Input image path
        output_path: Output image path
        config: Processing configuration
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read {image_path}")
        return
    
    # Apply edge noise (professor's main requirement)
    if config.get("edge_noise", True):
        image = add_edge_noise(image, strength=config.get("edge_noise_strength", 0.02))
    
    # Apply general image noise
    if config.get("image_noise", True):
        image = add_gaussian_noise(image, strength=config.get("image_noise_strength", 0.01))
    
    # Apply color jitter
    if config.get("color_jitter", True):
        image = apply_color_jitter(
            image,
            brightness_range=config.get("brightness_range", (-0.1, 0.1)),
            contrast_range=config.get("contrast_range", (0.9, 1.1)),
            saturation_range=config.get("saturation_range", (0.9, 1.1))
        )
    
    # Apply subtle chromatic aberration (optional)
    if config.get("chromatic_aberration", False):
        image = add_chromatic_aberration(image, shift=1)
    
    # Apply very subtle motion blur (optional, simulates handheld camera)
    if config.get("motion_blur", False):
        if np.random.random() < 0.3:  # 30% chance
            angle = np.random.uniform(0, 360)
            image = add_motion_blur(image, kernel_size=3, angle=angle)
    
    # Save processed image
    cv2.imwrite(output_path, image)


def process_directory(input_dir: str, output_dir: str = None, config: dict = None) -> None:
    """
    Process all images in a directory.
    
    Args:
        input_dir: Input directory with images
        output_dir: Output directory (None = overwrite originals)
        config: Processing configuration
    """
    if config is None:
        config = {
            "edge_noise": True,
            "edge_noise_strength": 0.02,
            "image_noise": True,
            "image_noise_strength": 0.01,
            "color_jitter": True,
            "brightness_range": (-0.05, 0.05),
            "contrast_range": (0.95, 1.05),
            "saturation_range": (0.95, 1.05),
        }
    
    if output_dir is None:
        output_dir = input_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    input_path = Path(input_dir)
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    print(f"Processing {len(image_files)} images...")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  - {key}: {value}")
    print()
    
    # Process each image
    for img_file in tqdm(image_files, desc="Post-processing"):
        output_file = Path(output_dir) / img_file.name
        process_image(str(img_file), str(output_file), config)
    
    print(f"\nDone! Processed {len(image_files)} images.")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Post-process synthetic images to add realistic noise and imperfections"
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input directory containing images"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory (default: overwrite originals)"
    )
    
    parser.add_argument(
        "--noise-level", "-n",
        type=float,
        default=0.02,
        help="Overall noise level (default: 0.02)"
    )
    
    parser.add_argument(
        "--edge-noise", "-e",
        type=float,
        default=0.03,
        help="Edge noise strength (default: 0.03)"
    )
    
    parser.add_argument(
        "--no-color-jitter",
        action="store_true",
        help="Disable color jitter"
    )
    
    parser.add_argument(
        "--chromatic-aberration",
        action="store_true",
        help="Enable chromatic aberration effect"
    )
    
    parser.add_argument(
        "--motion-blur",
        action="store_true",
        help="Enable random motion blur"
    )
    
    args = parser.parse_args()
    
    # Build config from arguments
    config = {
        "edge_noise": True,
        "edge_noise_strength": args.edge_noise,
        "image_noise": True,
        "image_noise_strength": args.noise_level,
        "color_jitter": not args.no_color_jitter,
        "brightness_range": (-0.05, 0.05),
        "contrast_range": (0.95, 1.05),
        "saturation_range": (0.95, 1.05),
        "chromatic_aberration": args.chromatic_aberration,
        "motion_blur": args.motion_blur,
    }
    
    # Process images
    process_directory(args.input, args.output, config)


if __name__ == "__main__":
    main()

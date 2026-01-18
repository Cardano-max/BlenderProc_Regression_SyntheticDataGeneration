"""
BlenderProc Synthetic Data Generation for Shackle Dimension Regression
======================================================================
Generates synthetic training data for predicting shackle HEIGHT and WIDTH.

The script:
1. Varies shackle dimensions (height, width) parametrically
2. Randomizes camera angles, lighting, backgrounds
3. Saves images with dimension labels for regression training

Author: Maidah Binte Tariq
Date: January 2025
Objective: Sim-to-Real Regression for Industrial Shackle Dimensions
"""

import blenderproc as bproc
import numpy as np
import argparse
import os
import json
import csv
import bpy
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

# Dimension ranges for parameterization (in mm or Blender units)
DIMENSION_CONFIG = {
    # Height variation range (percentage of original)
    "height_scale_range": [0.8, 1.2],   # 80% to 120% of original height
    
    # Width variation range (percentage of original)  
    "width_scale_range": [0.8, 1.2],    # 80% to 120% of original width
    
    # You can also use absolute values if known:
    # "height_range_mm": [100, 150],    # Height in mm
    # "width_range_mm": [50, 80],       # Width in mm
}

CAMERA_CONFIG = {
    "lens_mm": 50,
    "distance_range": [400, 800],
    "elevation_range": [20, 60],      # Degrees from horizontal
    "azimuth_range": [0, 360],        # Full rotation
    "dof_fstop": 2.8,
}

LIGHTING_CONFIG = {
    "hdri_rotation_range": [0, 360],  # Random HDRI rotation
    "hdri_strength_range": [0.5, 1.5],
    "sun_energy_range": [1.0, 5.0],
    "sun_elevation_range": [30, 70],
}


# ============================================================================
# DIMENSION MANIPULATION FUNCTIONS
# ============================================================================

def get_object_dimensions(obj):
    """Get the current dimensions of a Blender object."""
    # Get bounding box dimensions
    bbox = obj.bound_box
    xs = [v[0] for v in bbox]
    ys = [v[1] for v in bbox]
    zs = [v[2] for v in bbox]
    
    width = max(xs) - min(xs)
    depth = max(ys) - min(ys)
    height = max(zs) - min(zs)
    
    return {
        "width": width * obj.scale[0],
        "depth": depth * obj.scale[1],
        "height": height * obj.scale[2],
    }


def set_shackle_dimensions(objects, height_scale, width_scale):
    """
    Scale shackle objects to achieve target height and width.
    
    Args:
        objects: List of Blender objects (shackle parts)
        height_scale: Scale factor for height (Z axis)
        width_scale: Scale factor for width (X axis)
    
    Returns:
        dict: Actual dimensions after scaling
    """
    for obj in objects:
        blender_obj = obj.blender_obj if hasattr(obj, 'blender_obj') else obj
        
        # Store original scale
        orig_scale = list(blender_obj.scale)
        
        # Apply new scale
        # Assuming: X = width, Y = depth, Z = height
        blender_obj.scale[0] = orig_scale[0] * width_scale   # Width (X)
        blender_obj.scale[2] = orig_scale[2] * height_scale  # Height (Z)
        # Keep depth (Y) unchanged or scale proportionally
        # blender_obj.scale[1] = orig_scale[1] * width_scale  # Uncomment if needed
        
    # Return the actual dimensions
    if objects:
        main_obj = objects[0].blender_obj if hasattr(objects[0], 'blender_obj') else objects[0]
        dims = get_object_dimensions(main_obj)
        dims["height_scale"] = height_scale
        dims["width_scale"] = width_scale
        return dims
    
    return {"height_scale": height_scale, "width_scale": width_scale}


def reset_object_scale(objects):
    """Reset objects to original scale (1, 1, 1)."""
    for obj in objects:
        blender_obj = obj.blender_obj if hasattr(obj, 'blender_obj') else obj
        blender_obj.scale = (1.0, 1.0, 1.0)


def sample_dimensions():
    """Sample random height and width scale factors."""
    height_scale = np.random.uniform(*DIMENSION_CONFIG["height_scale_range"])
    width_scale = np.random.uniform(*DIMENSION_CONFIG["width_scale_range"])
    return height_scale, width_scale


# ============================================================================
# SCENE SETUP FUNCTIONS
# ============================================================================

def load_shackle_model(blend_file: str):
    """Load shackle model from blend file."""
    print(f"Loading model from: {blend_file}")
    objs = bproc.loader.load_blend(blend_file)
    mesh_objs = [obj for obj in objs if isinstance(obj, bproc.types.MeshObject)]
    
    for obj in mesh_objs:
        obj.set_cp("category_id", 1)
    
    print(f"Loaded {len(mesh_objs)} mesh objects")
    return mesh_objs


def get_object_center(objects):
    """Calculate center of all objects."""
    if not objects:
        return np.array([0, 0, 100])
    locations = [obj.get_location() for obj in objects]
    return np.mean(locations, axis=0)


def setup_random_camera(target: np.ndarray):
    """Setup camera with randomized position."""
    distance = np.random.uniform(*CAMERA_CONFIG["distance_range"])
    elevation = np.radians(np.random.uniform(*CAMERA_CONFIG["elevation_range"]))
    azimuth = np.radians(np.random.uniform(*CAMERA_CONFIG["azimuth_range"]))
    
    x = distance * np.cos(elevation) * np.cos(azimuth)
    y = distance * np.cos(elevation) * np.sin(azimuth)
    z = distance * np.sin(elevation)
    
    camera_pos = target + np.array([x, y, z])
    
    rotation = bproc.camera.rotation_from_forward_vec(
        target - camera_pos,
        inplane_rot=np.random.uniform(-0.05, 0.05)
    )
    
    cam2world = bproc.math.build_transformation_mat(camera_pos, rotation)
    bproc.camera.add_camera_pose(cam2world)
    bproc.camera.set_intrinsics_from_blender_params(lens=CAMERA_CONFIG["lens_mm"], lens_unit="MILLIMETERS")
    
    # Setup DOF
    cam = bpy.context.scene.camera
    if cam:
        cam.data.dof.use_dof = True
        cam.data.dof.focus_distance = distance
        cam.data.dof.aperture_fstop = CAMERA_CONFIG["dof_fstop"]
    
    return {"distance": distance, "elevation": np.degrees(elevation), "azimuth": np.degrees(azimuth)}


def setup_lighting(hdri_path: str = None):
    """Setup randomized lighting."""
    # Rotate HDRI
    hdri_rotation = np.random.uniform(*LIGHTING_CONFIG["hdri_rotation_range"])
    hdri_strength = np.random.uniform(*LIGHTING_CONFIG["hdri_strength_range"])
    
    world = bpy.context.scene.world
    if world and world.use_nodes:
        for node in world.node_tree.nodes:
            if node.type == 'MAPPING':
                node.inputs['Rotation'].default_value[2] = np.radians(hdri_rotation)
            if node.type == 'BACKGROUND':
                node.inputs['Strength'].default_value = hdri_strength
    
    # Add sun
    sun_energy = np.random.uniform(*LIGHTING_CONFIG["sun_energy_range"])
    sun_elevation = np.radians(np.random.uniform(*LIGHTING_CONFIG["sun_elevation_range"]))
    sun_azimuth = np.radians(np.random.uniform(0, 360))
    
    sun = bproc.types.Light()
    sun.set_type("SUN")
    sun.set_energy(sun_energy)
    
    sun_dir = np.array([
        np.cos(sun_elevation) * np.cos(sun_azimuth),
        np.cos(sun_elevation) * np.sin(sun_azimuth),
        np.sin(sun_elevation)
    ])
    sun.set_location(sun_dir * 1000)
    
    return {
        "hdri_rotation": hdri_rotation,
        "hdri_strength": hdri_strength,
        "sun_energy": sun_energy,
        "sun_elevation": np.degrees(sun_elevation),
        "sun_azimuth": np.degrees(sun_azimuth),
    }


# ============================================================================
# RENDERING AND DATA SAVING
# ============================================================================

def render_and_save(output_dir: str, index: int, dimensions: dict, camera_params: dict, light_params: dict):
    """Render frame and save with metadata."""
    
    # Create filename with dimension info
    h_scale = dimensions.get("height_scale", 1.0)
    w_scale = dimensions.get("width_scale", 1.0)
    filename = f"shackle_h{h_scale:.3f}_w{w_scale:.3f}_{index:06d}"
    
    # Render
    data = bproc.renderer.render()
    
    # Save image
    if "colors" in data:
        import cv2
        rgb = data["colors"][0]
        img_path = os.path.join(output_dir, "images", f"{filename}.png")
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        cv2.imwrite(img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    
    # Return metadata for labels
    return {
        "filename": f"{filename}.png",
        "index": index,
        "height_scale": h_scale,
        "width_scale": w_scale,
        "height": dimensions.get("height", 0),
        "width": dimensions.get("width", 0),
        "depth": dimensions.get("depth", 0),
        "camera": camera_params,
        "lighting": light_params,
    }


def save_regression_labels(metadata_list: list, output_dir: str):
    """Save labels for regression training in multiple formats."""
    
    # === CSV Format (Simple, works with most ML frameworks) ===
    csv_path = os.path.join(output_dir, "labels.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow([
            "filename", 
            "height_scale", 
            "width_scale",
            "height_mm",
            "width_mm",
            "depth_mm",
            "camera_distance",
            "camera_elevation",
            "camera_azimuth"
        ])
        # Data
        for meta in metadata_list:
            writer.writerow([
                meta["filename"],
                meta["height_scale"],
                meta["width_scale"],
                meta.get("height", 0),
                meta.get("width", 0),
                meta.get("depth", 0),
                meta["camera"]["distance"],
                meta["camera"]["elevation"],
                meta["camera"]["azimuth"],
            ])
    print(f"Saved CSV labels: {csv_path}")
    
    # === JSON Format (Full metadata) ===
    json_path = os.path.join(output_dir, "labels.json")
    with open(json_path, 'w') as f:
        json.dump({
            "description": "Shackle Dimension Regression Dataset",
            "targets": ["height_scale", "width_scale"],
            "num_samples": len(metadata_list),
            "dimension_ranges": DIMENSION_CONFIG,
            "samples": metadata_list,
        }, f, indent=2)
    print(f"Saved JSON labels: {json_path}")
    
    # === PyTorch/TensorFlow Format (image_path, labels) ===
    txt_path = os.path.join(output_dir, "labels.txt")
    with open(txt_path, 'w') as f:
        f.write("# filename height_scale width_scale\n")
        for meta in metadata_list:
            f.write(f"{meta['filename']} {meta['height_scale']:.4f} {meta['width_scale']:.4f}\n")
    print(f"Saved TXT labels: {txt_path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="BlenderProc Shackle Dimension Regression Data Generation")
    parser.add_argument("blend_file", type=str, help="Path to .blend file")
    parser.add_argument("num_images", type=int, nargs="?", default=10, help="Number of images (default: 10)")
    parser.add_argument("--output", "-o", type=str, default="./regression_output", help="Output directory")
    parser.add_argument("--hdri", type=str, default=None, help="HDRI file path")
    parser.add_argument("--height-range", type=float, nargs=2, default=[0.8, 1.2], help="Height scale range")
    parser.add_argument("--width-range", type=float, nargs=2, default=[0.8, 1.2], help="Width scale range")
    parser.add_argument("--resolution", type=int, nargs=2, default=[1920, 1080], help="Resolution WxH")
    
    args = parser.parse_args()
    
    # Update config from args
    DIMENSION_CONFIG["height_scale_range"] = args.height_range
    DIMENSION_CONFIG["width_scale_range"] = args.width_range
    
    # Validate
    if not os.path.exists(args.blend_file):
        print(f"Error: Blend file not found: {args.blend_file}")
        return
    
    # Setup output
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "images"), exist_ok=True)
    
    print("=" * 60)
    print("  BlenderProc Dimension Regression Data Generation")
    print("=" * 60)
    print(f"  Blend file: {args.blend_file}")
    print(f"  Images: {args.num_images}")
    print(f"  Output: {args.output}")
    print(f"  Height range: {args.height_range}")
    print(f"  Width range: {args.width_range}")
    print("=" * 60)
    
    all_metadata = []
    
    for i in range(args.num_images):
        print(f"\nGenerating image {i+1}/{args.num_images}...")
        
        # Initialize fresh scene
        bproc.clean_up()
        bproc.init()
        
        # Load model
        objects = load_shackle_model(args.blend_file)
        
        if not objects:
            print("Warning: No mesh objects found!")
            continue
        
        # === KEY: Sample and apply random dimensions ===
        height_scale, width_scale = sample_dimensions()
        dimensions = set_shackle_dimensions(objects, height_scale, width_scale)
        print(f"  Dimensions: height_scale={height_scale:.3f}, width_scale={width_scale:.3f}")
        
        # Get target for camera
        target = get_object_center(objects)
        
        # Setup camera
        camera_params = setup_random_camera(target)
        
        # Setup lighting
        light_params = setup_lighting(args.hdri)
        
        # Configure renderer
        bproc.camera.set_resolution(args.resolution[0], args.resolution[1])
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.samples = 128
        bpy.context.scene.cycles.use_denoising = True
        bpy.context.scene.view_settings.view_transform = 'Filmic'
        
        # Render and save
        try:
            metadata = render_and_save(args.output, i, dimensions, camera_params, light_params)
            all_metadata.append(metadata)
            print(f"  ✓ Saved: {metadata['filename']}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    # Save all labels
    save_regression_labels(all_metadata, args.output)
    
    print("\n" + "=" * 60)
    print("  Generation Complete!")
    print(f"  Images: {len(all_metadata)}")
    print(f"  Output: {args.output}")
    print(f"  Labels: labels.csv, labels.json, labels.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()

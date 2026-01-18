"""
Simplified BlenderProc Data Generation Script
=============================================
A simpler version for quick testing and initial data generation.

Usage:
    blenderproc run simple_pipeline.py path/to/model.blend 50

Author: Maidah Binte Tariq
Date: January 2025
"""

import blenderproc as bproc
import numpy as np
import argparse
import os
import sys
import bpy
from pathlib import Path


def setup_scene(blend_file: str) -> list:
    """Load the blend file and return mesh objects."""
    print(f"Loading: {blend_file}")
    
    # Load all objects from blend file
    objs = bproc.loader.load_blend(blend_file)
    
    # Filter for mesh objects
    mesh_objs = [obj for obj in objs if isinstance(obj, bproc.types.MeshObject)]
    
    # Set category ID for segmentation
    for i, obj in enumerate(mesh_objs):
        obj.set_cp("category_id", 1)
    
    print(f"Loaded {len(mesh_objs)} mesh objects")
    return mesh_objs


def get_object_center(objects: list) -> np.ndarray:
    """Calculate the center of all objects."""
    if not objects:
        return np.array([0, 0, 100])
    
    locations = [obj.get_location() for obj in objects]
    return np.mean(locations, axis=0)


def setup_random_camera(target: np.ndarray, 
                        distance_range=(400, 800),
                        elevation_range=(20, 60)) -> None:
    """Setup camera with randomized spherical positioning."""
    # Random spherical coordinates
    distance = np.random.uniform(*distance_range)
    elevation = np.radians(np.random.uniform(*elevation_range))
    azimuth = np.radians(np.random.uniform(0, 360))
    
    # Convert to Cartesian
    x = distance * np.cos(elevation) * np.cos(azimuth)
    y = distance * np.cos(elevation) * np.sin(azimuth)
    z = distance * np.sin(elevation)
    
    camera_pos = target + np.array([x, y, z])
    
    # Look at target
    rotation = bproc.camera.rotation_from_forward_vec(
        target - camera_pos,
        inplane_rot=np.random.uniform(-0.05, 0.05)
    )
    
    # Set camera pose
    cam2world = bproc.math.build_transformation_mat(camera_pos, rotation)
    bproc.camera.add_camera_pose(cam2world)
    
    # Set lens
    bproc.camera.set_intrinsics_from_blender_params(lens=50, lens_unit="MILLIMETERS")
    
    # Setup DOF for background blur
    cam = bpy.context.scene.camera
    if cam:
        cam.data.dof.use_dof = True
        cam.data.dof.focus_distance = distance
        cam.data.dof.aperture_fstop = 2.8


def setup_lighting(hdri_path: str = None) -> None:
    """Setup HDRI lighting with random rotation."""
    if hdri_path and os.path.exists(hdri_path):
        bproc.world.set_world_background_hdr_img(hdri_path)
    
    # Rotate HDRI randomly
    world = bpy.context.scene.world
    if world and world.use_nodes:
        for node in world.node_tree.nodes:
            if node.type == 'MAPPING':
                node.inputs['Rotation'].default_value[2] = np.radians(np.random.uniform(0, 360))
            if node.type == 'BACKGROUND':
                node.inputs['Strength'].default_value = np.random.uniform(0.5, 1.5)
    
    # Add sun for extra variation
    sun = bproc.types.Light()
    sun.set_type("SUN")
    sun.set_energy(np.random.uniform(1, 4))
    
    # Random sun direction
    sun_elev = np.radians(np.random.uniform(30, 70))
    sun_azim = np.radians(np.random.uniform(0, 360))
    sun_dir = np.array([
        np.cos(sun_elev) * np.cos(sun_azim),
        np.cos(sun_elev) * np.sin(sun_azim),
        np.sin(sun_elev)
    ])
    sun.set_location(sun_dir * 1000)


def render_frame(output_dir: str, index: int, prefix: str = "shackle") -> dict:
    """Render a single frame and save outputs."""
    # Enable segmentation
    bproc.renderer.enable_segmentation_output(map_by=["category_id"])
    
    # Render
    data = bproc.renderer.render()
    
    # Save outputs
    filename = f"{prefix}_{index:06d}"
    
    # Save using HDF5 (BlenderProc's native format)
    bproc.writer.write_hdf5(output_dir, data, append_to_existing_output=True)
    
    # Also save as PNG
    if "colors" in data:
        import cv2
        rgb = data["colors"][0]
        img_path = os.path.join(output_dir, f"{filename}.png")
        cv2.imwrite(img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    
    return {"filename": f"{filename}.png", "index": index}


def main():
    parser = argparse.ArgumentParser(description="Simple BlenderProc data generation")
    parser.add_argument("blend_file", type=str, help="Path to .blend file")
    parser.add_argument("num_images", type=int, nargs="?", default=10, help="Number of images (default: 10)")
    parser.add_argument("--output", "-o", type=str, default="./simple_output", help="Output directory")
    parser.add_argument("--hdri", type=str, default=None, help="HDRI file path")
    parser.add_argument("--resolution", type=int, nargs=2, default=[1920, 1080], help="Resolution WxH")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.blend_file):
        print(f"Error: Blend file not found: {args.blend_file}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("="*60)
    print("  BlenderProc Simple Data Generation")
    print("="*60)
    print(f"  Blend file: {args.blend_file}")
    print(f"  Images: {args.num_images}")
    print(f"  Output: {args.output}")
    print(f"  Resolution: {args.resolution[0]}x{args.resolution[1]}")
    print("="*60)
    
    # Main generation loop
    for i in range(args.num_images):
        print(f"\nGenerating image {i+1}/{args.num_images}...")
        
        # Initialize fresh scene
        bproc.clean_up()
        bproc.init()
        
        # Load model
        objects = setup_scene(args.blend_file)
        
        if not objects:
            print("Warning: No mesh objects found!")
            continue
        
        # Get target location
        target = get_object_center(objects)
        
        # Setup camera
        setup_random_camera(target)
        
        # Setup lighting
        setup_lighting(args.hdri)
        
        # Configure renderer
        bproc.camera.set_resolution(args.resolution[0], args.resolution[1])
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.samples = 128
        bpy.context.scene.cycles.use_denoising = True
        bpy.context.scene.view_settings.view_transform = 'Filmic'
        
        # Render
        try:
            result = render_frame(args.output, i)
            print(f"  ✓ Saved: {result['filename']}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    print("\n" + "="*60)
    print("  Generation Complete!")
    print(f"  Output: {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()

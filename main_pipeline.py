"""
BlenderProc Synthetic Data Generation Pipeline
================================================
Sim-to-Real Synthetic Data for Industrial Shackle Wear Detection

Author: Maidah Binte Tariq
Project: Thesis - Wear Detection using Synthetic Data
Date: January 2025

This script generates photorealistic synthetic training data with:
- Domain randomization (camera, lighting, texture)
- Blurred real-world backgrounds
- COCO-format annotations for YOLO training
"""

import blenderproc as bproc
import numpy as np
import argparse
import yaml
import os
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import bpy


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    "num_images": 100,
    "resolution": {"width": 1920, "height": 1080},
    "wear_levels": [
        {"level": 0.00, "weight": 0.15, "blend_file": "Shackle_00_Clean.blend"},
        {"level": 0.25, "weight": 0.25, "blend_file": "Shackle_25_Moderate.blend"},
        {"level": 0.50, "weight": 0.35, "blend_file": "Shackle_50_Heavy.blend"},
        {"level": 0.75, "weight": 0.25, "blend_file": "Shackle_75_Severe.blend"},
    ],
    "camera": {
        "lens_mm": 50,
        "distance_range": [400, 800],
        "elevation_range": [20, 60],
        "azimuth_range": [0, 360],
        "dof_fstop": 2.8,
        "dof_enabled": True,
    },
    "lighting": {
        "hdri_strength_range": [0.5, 1.5],
        "sun_energy_range": [1.0, 5.0],
        "sun_elevation_range": [30, 80],
        "sun_azimuth_range": [0, 360],
    },
    "materials": {
        "color_variation": True,
        "rust_hue_range": [0.02, 0.08],  # Orange-brown range
        "rust_saturation_range": [0.6, 0.9],
        "rust_value_range": [0.3, 0.6],
    },
    "output": {
        "format": "PNG",
        "save_depth": False,
        "save_normals": False,
        "save_segmentation": True,
    },
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file or use defaults."""
    config = DEFAULT_CONFIG.copy()
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
            # Deep merge user config into default
            def merge_dict(base, update):
                for key, value in update.items():
                    if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                        merge_dict(base[key], value)
                    else:
                        base[key] = value
            merge_dict(config, user_config)
    
    return config


def setup_output_directories(base_path: str) -> Dict[str, str]:
    """Create output directory structure."""
    dirs = {
        "images": os.path.join(base_path, "images"),
        "annotations": os.path.join(base_path, "annotations"),
        "masks": os.path.join(base_path, "masks"),
        "depth": os.path.join(base_path, "depth"),
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def sample_wear_level(config: dict) -> Tuple[float, str]:
    """Sample a wear level based on configured weights."""
    levels = config["wear_levels"]
    weights = [l["weight"] for l in levels]
    weights = np.array(weights) / sum(weights)  # Normalize
    
    idx = np.random.choice(len(levels), p=weights)
    selected = levels[idx]
    
    return selected["level"], selected.get("blend_file", None)


# ============================================================================
# SCENE SETUP FUNCTIONS
# ============================================================================

def load_shackle_model(blend_file_path: str) -> List[bproc.types.MeshObject]:
    """Load shackle model from blend file."""
    print(f"Loading model from: {blend_file_path}")
    
    # Load the blend file
    objs = bproc.loader.load_blend(blend_file_path)
    
    # Filter for mesh objects only
    mesh_objs = [obj for obj in objs if isinstance(obj, bproc.types.MeshObject)]
    
    # Set object properties for rendering
    for obj in mesh_objs:
        obj.set_cp("category_id", 1)  # For segmentation
        obj.enable_rigidbody(False)
    
    print(f"Loaded {len(mesh_objs)} mesh objects")
    return mesh_objs


def load_shackle_from_stl(stl_path: str, material_path: Optional[str] = None) -> List[bproc.types.MeshObject]:
    """Load shackle from STL file and optionally apply material."""
    print(f"Loading STL from: {stl_path}")
    
    objs = bproc.loader.load_obj(stl_path)
    
    for obj in objs:
        obj.set_cp("category_id", 1)
        
        # Apply material if provided
        if material_path and os.path.exists(material_path):
            # Load material from blend file
            bproc.loader.load_blend(material_path, obj_types=['material'])
    
    return objs


def setup_camera(config: dict, target_location: np.ndarray) -> None:
    """Setup camera with randomized position around target."""
    cam_config = config["camera"]
    
    # Sample random spherical coordinates
    distance = np.random.uniform(*cam_config["distance_range"])
    elevation = np.random.uniform(*cam_config["elevation_range"])
    azimuth = np.random.uniform(*cam_config["azimuth_range"])
    
    # Convert to Cartesian coordinates
    elevation_rad = np.radians(elevation)
    azimuth_rad = np.radians(azimuth)
    
    x = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
    y = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
    z = distance * np.sin(elevation_rad)
    
    camera_location = target_location + np.array([x, y, z])
    
    # Create rotation matrix to look at target
    rotation_matrix = bproc.camera.rotation_from_forward_vec(
        target_location - camera_location,
        inplane_rot=np.random.uniform(-0.1, 0.1)  # Slight roll variation
    )
    
    # Build camera pose matrix
    cam2world_matrix = bproc.math.build_transformation_mat(camera_location, rotation_matrix)
    
    # Add camera pose
    bproc.camera.add_camera_pose(cam2world_matrix)
    
    # Set camera intrinsics
    bproc.camera.set_intrinsics_from_blender_params(
        lens=cam_config["lens_mm"],
        lens_unit="MILLIMETERS"
    )
    
    # Setup depth of field
    if cam_config.get("dof_enabled", True):
        setup_depth_of_field(cam_config, target_location, camera_location)


def setup_depth_of_field(cam_config: dict, target: np.ndarray, camera_pos: np.ndarray) -> None:
    """Configure camera depth of field for blurred background."""
    # Access Blender camera directly
    cam = bpy.context.scene.camera
    if cam:
        cam.data.dof.use_dof = True
        cam.data.dof.focus_distance = np.linalg.norm(target - camera_pos)
        cam.data.dof.aperture_fstop = cam_config.get("dof_fstop", 2.8)


def setup_lighting(config: dict) -> None:
    """Setup HDRI environment and optional sun light with randomization."""
    light_config = config["lighting"]
    
    # Random HDRI strength
    hdri_strength = np.random.uniform(*light_config["hdri_strength_range"])
    
    # Set world background strength
    world = bpy.context.scene.world
    if world and world.use_nodes:
        for node in world.node_tree.nodes:
            if node.type == 'BACKGROUND':
                node.inputs['Strength'].default_value = hdri_strength
    
    # Add randomized sun light for additional variation
    sun_energy = np.random.uniform(*light_config["sun_energy_range"])
    sun_elevation = np.random.uniform(*light_config["sun_elevation_range"])
    sun_azimuth = np.random.uniform(*light_config["sun_azimuth_range"])
    
    # Create sun light
    sun = bproc.types.Light()
    sun.set_type("SUN")
    sun.set_energy(sun_energy)
    
    # Set sun direction
    sun_dir = np.array([
        np.cos(np.radians(sun_elevation)) * np.cos(np.radians(sun_azimuth)),
        np.cos(np.radians(sun_elevation)) * np.sin(np.radians(sun_azimuth)),
        np.sin(np.radians(sun_elevation))
    ])
    sun.set_location(sun_dir * 1000)  # Far away sun


def load_hdri_background(hdri_path: str, rotation: float = 0) -> None:
    """Load HDRI environment map."""
    if os.path.exists(hdri_path):
        bproc.world.set_world_background_hdr_img(hdri_path)
        
        # Rotate HDRI for variation
        world = bpy.context.scene.world
        if world and world.use_nodes:
            for node in world.node_tree.nodes:
                if node.type == 'MAPPING':
                    node.inputs['Rotation'].default_value[2] = np.radians(rotation)


def load_background_image(bg_path: str) -> None:
    """Load a real background image as world background."""
    if os.path.exists(bg_path):
        # Create world nodes for background image
        world = bpy.context.scene.world
        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links
        
        # Clear existing nodes
        nodes.clear()
        
        # Create nodes
        output = nodes.new('ShaderNodeOutputWorld')
        background = nodes.new('ShaderNodeBackground')
        tex_image = nodes.new('ShaderNodeTexEnvironment')
        
        # Load image
        tex_image.image = bpy.data.images.load(bg_path)
        
        # Connect nodes
        links.new(tex_image.outputs['Color'], background.inputs['Color'])
        links.new(background.outputs['Background'], output.inputs['Surface'])


# ============================================================================
# MATERIAL RANDOMIZATION
# ============================================================================

def randomize_rust_color(config: dict, objects: List[bproc.types.MeshObject]) -> None:
    """Apply random color variation to rust materials."""
    if not config["materials"].get("color_variation", False):
        return
    
    mat_config = config["materials"]
    
    for obj in objects:
        for mat_slot in obj.blender_obj.material_slots:
            mat = mat_slot.material
            if mat and mat.use_nodes:
                # Find Principled BSDF nodes
                for node in mat.node_tree.nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        # Check if this is a rust shader (low metallic)
                        if node.inputs['Metallic'].default_value < 0.5:
                            # Randomize rust color
                            hue = np.random.uniform(*mat_config["rust_hue_range"])
                            sat = np.random.uniform(*mat_config["rust_saturation_range"])
                            val = np.random.uniform(*mat_config["rust_value_range"])
                            
                            # Convert HSV to RGB
                            import colorsys
                            r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
                            
                            node.inputs['Base Color'].default_value = (r, g, b, 1.0)


def set_wear_level(objects: List[bproc.types.MeshObject], wear_level: float) -> None:
    """Set wear level on material's Wear_Level value node."""
    for obj in objects:
        for mat_slot in obj.blender_obj.material_slots:
            mat = mat_slot.material
            if mat and mat.use_nodes:
                for node in mat.node_tree.nodes:
                    if node.type == 'VALUE' and 'wear' in node.label.lower():
                        node.outputs[0].default_value = wear_level


# ============================================================================
# RENDERING FUNCTIONS
# ============================================================================

def setup_renderer(config: dict) -> None:
    """Configure Cycles renderer settings."""
    resolution = config["resolution"]
    
    # Set resolution
    bproc.camera.set_resolution(resolution["width"], resolution["height"])
    
    # Configure Cycles
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.samples = 128
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    
    # Adaptive sampling
    bpy.context.scene.cycles.use_adaptive_sampling = True
    bpy.context.scene.cycles.adaptive_threshold = 0.005
    
    # Color management
    bpy.context.scene.view_settings.view_transform = 'Filmic'
    bpy.context.scene.view_settings.look = 'Medium Contrast'


def render_scene(output_dirs: Dict[str, str], image_index: int, wear_level: float, config: dict) -> Dict:
    """Render the scene and save outputs."""
    # Generate filename
    wear_str = f"{int(wear_level * 100):03d}"
    filename = f"shackle_{wear_str}_{image_index:06d}"
    
    # Enable render passes
    bproc.renderer.enable_segmentation_output(map_by=["category_id"])
    
    if config["output"].get("save_depth", False):
        bproc.renderer.enable_depth_output(activate_antialiasing=False)
    
    if config["output"].get("save_normals", False):
        bproc.renderer.enable_normals_output()
    
    # Render
    data = bproc.renderer.render()
    
    # Save outputs
    image_path = os.path.join(output_dirs["images"], f"{filename}.png")
    
    # Save RGB image
    bproc.writer.write_hdf5(
        output_dirs["images"],
        data,
        append_to_existing_output=True
    )
    
    # Alternative: Save as individual files
    import cv2
    if "colors" in data:
        rgb = data["colors"][0]
        cv2.imwrite(image_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    
    # Save segmentation mask
    if "category_id_segmaps" in data:
        mask_path = os.path.join(output_dirs["masks"], f"{filename}_mask.png")
        mask = data["category_id_segmaps"][0]
        cv2.imwrite(mask_path, mask.astype(np.uint8) * 255)
    
    # Return metadata for annotation
    return {
        "filename": f"{filename}.png",
        "wear_level": wear_level,
        "image_path": image_path,
    }


# ============================================================================
# ANNOTATION FUNCTIONS
# ============================================================================

def create_coco_annotation(image_info: Dict, image_id: int, objects: List) -> Dict:
    """Create COCO format annotation for an image."""
    # Get bounding box from rendered mask or object bounds
    annotations = []
    
    for obj_id, obj in enumerate(objects):
        # Get 2D bounding box
        bbox = get_2d_bbox(obj)
        
        if bbox:
            x, y, w, h = bbox
            
            annotation = {
                "id": image_id * 100 + obj_id,
                "image_id": image_id,
                "category_id": get_category_from_wear(image_info["wear_level"]),
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
                "segmentation": [],  # Can add polygon if needed
            }
            annotations.append(annotation)
    
    return annotations


def get_2d_bbox(obj: bproc.types.MeshObject) -> Optional[Tuple[int, int, int, int]]:
    """Calculate 2D bounding box of object in rendered image."""
    # This is a simplified version - in practice, use rendered masks
    try:
        # Get object's 3D bounding box corners
        bbox_3d = obj.get_bound_box()
        
        # Project to 2D (requires camera matrix)
        # For now, return estimated bbox
        return None  # Implement based on your needs
    except:
        return None


def get_category_from_wear(wear_level: float) -> int:
    """Map wear level to category ID."""
    if wear_level < 0.125:
        return 1  # clean
    elif wear_level < 0.375:
        return 2  # worn_25
    elif wear_level < 0.625:
        return 3  # worn_50
    else:
        return 4  # worn_75


def save_coco_annotations(annotations: List[Dict], output_path: str, config: dict) -> None:
    """Save annotations in COCO format."""
    coco_data = {
        "info": {
            "description": "Synthetic Shackle Wear Detection Dataset",
            "version": "1.0",
            "year": 2025,
            "contributor": "Maidah Binte Tariq",
            "date_created": "2025-01-18",
        },
        "licenses": [
            {"id": 1, "name": "Research Use Only", "url": ""}
        ],
        "categories": [
            {"id": 1, "name": "shackle_clean", "supercategory": "shackle"},
            {"id": 2, "name": "shackle_worn_25", "supercategory": "shackle"},
            {"id": 3, "name": "shackle_worn_50", "supercategory": "shackle"},
            {"id": 4, "name": "shackle_worn_75", "supercategory": "shackle"},
        ],
        "images": [],
        "annotations": [],
    }
    
    for i, ann in enumerate(annotations):
        # Add image info
        coco_data["images"].append({
            "id": i,
            "file_name": ann["filename"],
            "width": config["resolution"]["width"],
            "height": config["resolution"]["height"],
        })
        
        # Add annotations (if available)
        if "annotations" in ann:
            coco_data["annotations"].extend(ann["annotations"])
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"Saved COCO annotations to: {output_path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main pipeline execution."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="BlenderProc Shackle Data Generation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--num-images", type=int, default=None, help="Override number of images")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--models-dir", type=str, default="./assets/models", help="Models directory")
    parser.add_argument("--hdri-dir", type=str, default="./assets/hdri", help="HDRI directory")
    parser.add_argument("--backgrounds-dir", type=str, default="./assets/backgrounds", help="Backgrounds directory")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    if args.num_images:
        config["num_images"] = args.num_images
    
    # Initialize BlenderProc
    bproc.init()
    
    # Setup output directories
    output_dirs = setup_output_directories(args.output_dir)
    
    # Setup renderer
    setup_renderer(config)
    
    # Get list of available HDRIs
    hdri_files = list(Path(args.hdri_dir).glob("*.hdr")) + list(Path(args.hdri_dir).glob("*.exr"))
    
    # Get list of background images
    bg_files = list(Path(args.backgrounds_dir).glob("*.jpg")) + list(Path(args.backgrounds_dir).glob("*.png"))
    
    # Collect all annotations
    all_annotations = []
    
    print(f"\n{'='*60}")
    print(f"Starting synthetic data generation")
    print(f"Total images to generate: {config['num_images']}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # Main generation loop
    for i in range(config["num_images"]):
        print(f"\nGenerating image {i+1}/{config['num_images']}...")
        
        # Clean scene for fresh start
        bproc.clean_up()
        bproc.init()
        
        # Sample wear level and get corresponding blend file
        wear_level, blend_file = sample_wear_level(config)
        print(f"  Wear level: {wear_level*100:.0f}%")
        
        # Load shackle model
        blend_path = os.path.join(args.models_dir, blend_file) if blend_file else None
        if blend_path and os.path.exists(blend_path):
            objects = load_shackle_model(blend_path)
        else:
            # Fallback: use master file and set wear level
            master_file = os.path.join(args.models_dir, "Shackle_Master.blend")
            objects = load_shackle_model(master_file)
            set_wear_level(objects, wear_level)
        
        # Get target location (center of objects)
        if objects:
            locations = [obj.get_location() for obj in objects]
            target_location = np.mean(locations, axis=0)
        else:
            target_location = np.array([0, 0, 100])
        
        # Setup camera with randomization
        setup_camera(config, target_location)
        
        # Load random HDRI or background image
        if hdri_files and random.random() < 0.7:  # 70% chance HDRI
            hdri_path = random.choice(hdri_files)
            rotation = np.random.uniform(0, 360)
            load_hdri_background(str(hdri_path), rotation)
        elif bg_files:
            bg_path = random.choice(bg_files)
            load_background_image(str(bg_path))
        
        # Setup lighting with randomization
        setup_lighting(config)
        
        # Randomize material colors
        randomize_rust_color(config, objects)
        
        # Render and save
        try:
            image_info = render_scene(output_dirs, i, wear_level, config)
            
            # Create annotation
            annotation = create_coco_annotation(image_info, i, objects)
            image_info["annotations"] = annotation
            all_annotations.append(image_info)
            
            print(f"  ✓ Saved: {image_info['filename']}")
            
        except Exception as e:
            print(f"  ✗ Error rendering image {i}: {e}")
            continue
    
    # Save annotations
    annotations_path = os.path.join(output_dirs["annotations"], "coco_annotations.json")
    save_coco_annotations(all_annotations, annotations_path, config)
    
    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"Images generated: {len(all_annotations)}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

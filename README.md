# BlenderProc Synthetic Data Generation Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![BlenderProc](https://img.shields.io/badge/BlenderProc-2.7+-orange.svg)](https://github.com/DLR-RM/BlenderProc)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Sim-to-Real Synthetic Data for Industrial Shackle Wear Detection

**Project:** Thesis - Wear Detection using Synthetic Data  
**Author:** Maidah Binte Tariq  
**Institution:** [University Name]

---

## 📋 Overview

This pipeline generates photorealistic synthetic training data for industrial shackle wear detection using BlenderProc. It creates domain-randomized images with:
- Variable wear levels (0%-75%)
- Random camera angles (drone perspective)
- Random lighting conditions
- Blurred real-world backgrounds
- COCO-format annotations for YOLO training

---

## 🔧 System Requirements

### Hardware (Recommended)
- **GPU**: NVIDIA RTX 3060 or better (8GB+ VRAM)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB+ free space for generated data

### Software
- **OS**: Ubuntu 20.04+ / Windows 10+ / macOS 12+
- **Python**: 3.9 - 3.11
- **Blender**: 3.6+ (auto-installed by BlenderProc)

---

## 📦 Installation

### Step 1: Create Virtual Environment
```bash
# Create project directory
mkdir -p ~/blenderproc_shackle
cd ~/blenderproc_shackle

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
.\venv\Scripts\activate  # Windows
```

### Step 2: Install BlenderProc
```bash
pip install blenderproc
```

### Step 3: Install Additional Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download BlenderProc Resources (First Run)
```bash
# This downloads Blender and required resources
blenderproc run --help
```

---

## 📁 Project Structure

```
blenderproc_shackle/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config.yaml                  # Pipeline configuration
├── main_pipeline.py             # Main BlenderProc script
├── postprocess.py               # Post-processing (edge noise)
├── run_generation.sh            # Linux/macOS run script
├── run_generation.bat           # Windows run script
│
├── assets/
│   ├── models/                  # Shackle CAD models (.blend/.stl)
│   │   ├── Shackle_00_Clean.blend
│   │   ├── Shackle_25_Moderate.blend
│   │   ├── Shackle_50_Heavy.blend
│   │   └── Shackle_75_Severe.blend
│   │
│   ├── hdri/                    # HDRI environments
│   │   └── outdoor_field.hdr
│   │
│   ├── backgrounds/             # Real background crops (blurred)
│   │   ├── bg_001.jpg
│   │   ├── bg_002.jpg
│   │   └── ...
│   │
│   └── textures/                # Rust textures (optional)
│       └── rust_variations/
│
├── output/                      # Generated data
│   ├── images/                  # RGB images
│   ├── annotations/             # COCO JSON annotations
│   ├── masks/                   # Segmentation masks
│   └── depth/                   # Depth maps (optional)
│
└── logs/                        # Generation logs
```

---

## ⚙️ Configuration (config.yaml)

Edit `config.yaml` to customize the pipeline:

```yaml
# Number of images to generate
num_images: 1000

# Output resolution
resolution:
  width: 1920
  height: 1080

# Wear level distribution (probability weights)
wear_levels:
  - level: 0.00
    weight: 0.15
  - level: 0.25
    weight: 0.25
  - level: 0.50
    weight: 0.35
  - level: 0.75
    weight: 0.25

# Camera settings (drone perspective)
camera:
  lens_mm: 50
  distance_range: [400, 800]    # Distance from object
  elevation_range: [20, 60]      # Degrees from horizontal
  azimuth_range: [0, 360]        # Full rotation
  dof_fstop: 2.8                 # Depth of field

# Lighting variations
lighting:
  hdri_strength_range: [0.5, 1.5]
  sun_energy_range: [1.0, 5.0]
  sun_angle_range: [0, 360]
```

---

## 🚀 Running the Pipeline

### Method 1: Using Shell Script (Recommended)
```bash
# Linux/macOS
chmod +x run_generation.sh
./run_generation.sh

# Windows
run_generation.bat
```

### Method 2: Direct BlenderProc Command
```bash
blenderproc run main_pipeline.py --config config.yaml --num-images 1000
```

### Method 3: Python Script
```bash
python run_pipeline.py --num-images 1000 --output-dir ./output
```

---

## 📊 Output Format

### Images
- Format: PNG (lossless) or JPEG (compressed)
- Naming: `shackle_{wear_level}_{index:06d}.png`
- Example: `shackle_050_000001.png` (50% wear, image 1)

### Annotations (COCO Format)
```json
{
  "images": [...],
  "annotations": [...],
  "categories": [
    {"id": 1, "name": "shackle_clean"},
    {"id": 2, "name": "shackle_worn_25"},
    {"id": 3, "name": "shackle_worn_50"},
    {"id": 4, "name": "shackle_worn_75"}
  ]
}
```

### For YOLO Training
Run the conversion script:
```bash
python convert_to_yolo.py --input output/annotations --output output/yolo_labels
```

---

## 🔬 Professor's Requirements Checklist

Based on meeting transcript:

| Requirement | Implementation | Status |
|-------------|---------------|--------|
| Blurred backgrounds | Camera DOF + real bg crops | ✅ |
| Shape/silhouette focus | Clean geometry, varied angles | ✅ |
| Lighting variation | Random HDRI rotation + sun | ✅ |
| Camera angle variation | Spherical sampling | ✅ |
| Texture variation | Procedural + color randomization | ✅ |
| Edge softening | Post-process noise | ✅ |
| Mass generation | BlenderProc automation | ✅ |
| COCO annotations | Built-in export | ✅ |

---

## 🛠️ Troubleshooting

### GPU Not Detected
```bash
# Check CUDA availability
nvidia-smi

# Force CPU rendering (slower)
blenderproc run main_pipeline.py --device cpu
```

### Memory Issues
- Reduce resolution in config.yaml
- Process in smaller batches
- Close other applications

### Texture Issues
- Ensure texture paths are correct in .blend files
- Use packed textures: File > External Data > Pack All

---

## 📈 Recommended Workflow

1. **Test Run**: Generate 10 images to verify setup
   ```bash
   blenderproc run main_pipeline.py --num-images 10
   ```

2. **Visual Check**: Review output/images/ for quality

3. **Full Generation**: Generate 1000+ images
   ```bash
   blenderproc run main_pipeline.py --num-images 1000
   ```

4. **Post-Process**: Apply edge noise
   ```bash
   python postprocess.py --input output/images --noise-level 0.02
   ```

5. **Convert Labels**: For YOLO training
   ```bash
   python convert_to_yolo.py
   ```

---

## 📚 References

- BlenderProc Documentation: https://dlr-rm.github.io/BlenderProc/
- COCO Format: https://cocodataset.org/#format-data
- YOLO Training: https://docs.ultralytics.com/

---

## 📞 Support

For issues with this pipeline, contact:
- Student: Maidah Binte Tariq
- Supervisor: [Professor Name]

---

*Pipeline created: January 2025*
*Last updated: January 2025*

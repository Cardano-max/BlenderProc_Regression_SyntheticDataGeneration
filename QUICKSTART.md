# 🚀 QUICK START GUIDE
## BlenderProc Synthetic Data Generation for Shackle Wear Detection

---

## Step 1: Install BlenderProc (5 minutes)

```bash
# Create project folder
mkdir blenderproc_shackle
cd blenderproc_shackle

# Install BlenderProc (includes Blender automatically)
pip install blenderproc

# Verify installation
blenderproc run --help
```

---

## Step 2: Prepare Your Files

Create this folder structure:
```
blenderproc_shackle/
├── assets/
│   ├── models/
│   │   └── YourShackle.blend    ← Put your .blend file here
│   └── hdri/
│       └── outdoor.hdr          ← Optional: HDRI for lighting
├── output/                       ← Generated images go here
├── simple_pipeline.py            ← Copy from this package
└── main_pipeline.py              ← Copy from this package
```

---

## Step 3: Quick Test (Generate 10 Images)

```bash
# Simple test - just needs your blend file
blenderproc run simple_pipeline.py assets/models/YourShackle.blend 10 --output ./output
```

This will generate 10 images with:
- Random camera angles
- Random lighting
- Background blur (DOF)
- Denoised Cycles rendering

---

## Step 4: Full Production Run (1000+ Images)

```bash
# Edit config.yaml first to customize settings, then:
blenderproc run main_pipeline.py --num-images 1000 --output ./output
```

---

## Step 5: Post-Process for Realistic Edges

```bash
# Add noise to soften CG-perfect edges
python postprocess.py --input ./output --edge-noise 0.02
```

---

## Step 6: Convert to YOLO Format

```bash
# Convert COCO annotations to YOLO format
python convert_to_yolo.py --split
```

---

## 🎯 Professor's Key Requirements Checklist

| Requirement | Solution | Command/Setting |
|-------------|----------|-----------------|
| Blurred background | Camera DOF | `dof_fstop: 2.8` in config |
| Shape focus | Clean geometry | Use your CAD model |
| Lighting variation | Random HDRI + Sun | Built into pipeline |
| Camera angles | Spherical sampling | `elevation_range`, `azimuth_range` |
| Edge softening | Post-process noise | `postprocess.py --edge-noise 0.02` |
| Color variation | Material randomization | `color_variation: true` |
| Mass generation | BlenderProc automation | `--num-images 1000` |

---

## 📊 Expected Output

After running, you'll have:
```
output/
├── images/
│   ├── shackle_000_000001.png
│   ├── shackle_025_000002.png
│   └── ... (1000+ images)
├── annotations/
│   └── coco_annotations.json
├── yolo_labels/           (after conversion)
│   ├── shackle_000_000001.txt
│   └── ...
└── dataset/               (after splitting)
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

---

## ⚡ Tips for Faster Generation

1. **Use GPU**: Make sure CUDA/OptiX is enabled
2. **Lower samples for testing**: `--samples 64`
3. **Reduce resolution for testing**: `--resolution 960 540`
4. **Batch on HPC**: Split into multiple jobs

---

## 🆘 Troubleshooting

**"GPU not found"**
```bash
# Use CPU (slower)
blenderproc run simple_pipeline.py model.blend 10 --device cpu
```

**"Out of memory"**
- Reduce resolution
- Reduce samples
- Close other applications

**"Module not found"**
```bash
pip install opencv-python numpy pillow tqdm pyyaml
```

---

## 📝 Notes from Professor Meeting

1. **Texture perfectness**: "Real rust is more uniform, faded" - Use simpler textures
2. **Color matters less**: "Can even use blue" - Focus on shape/silhouette
3. **Edge sharpness**: "Never this crisp in real photos" - Add post-process noise
4. **Background**: "Use real image crops" - Can use HDRI or real photos
5. **Variation is key**: "Vary everything to avoid overfitting"

---

**Good luck with your thesis! 🎓**

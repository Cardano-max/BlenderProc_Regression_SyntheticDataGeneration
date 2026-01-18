# 📁 Assets Directory

This directory contains all the resources needed for synthetic data generation.

## Structure:

```
assets/
├── models/          # Blender shackle model files (.blend)
├── hdri/            # HDRI environment maps for lighting (.hdr, .exr)
└── backgrounds/     # Real background image crops (.jpg, .png)
```

## Setup Instructions:

### 1. Models (Required)
Add your shackle `.blend` files to `models/`:
- Download from your CAD software export
- Or use the provided master file

### 2. HDRI (Recommended)
Download free HDRIs from [Poly Haven](https://polyhaven.com/hdris):
```bash
cd assets/hdri
wget https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/alps_field_2k.hdr
```

### 3. Backgrounds (Optional but Recommended)
Add cropped, blurred background images from real photos.

## Why These Aren't Included in Git:
- **Large file sizes** - HDRIs and blend files are typically 50-500MB each
- **Copyright** - Some assets may have usage restrictions
- **Customization** - You should use your own project-specific assets

## Using Git LFS (Optional):
If you want to version control large files:
```bash
git lfs install
git lfs track "*.blend"
git lfs track "*.hdr"
git lfs track "*.exr"
```

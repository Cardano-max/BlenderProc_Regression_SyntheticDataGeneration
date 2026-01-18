#!/bin/bash
# ============================================================================
# Git Setup Script for BlenderProc Pipeline
# Repository: Cardano-max/BlenderProc_Regression_SyntheticDataGeneration
# ============================================================================

echo "============================================"
echo "  Setting up Git Repository"
echo "============================================"

# Initialize git repository
git init

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
pip-log.txt

# Blender
*.blend1
*.blend2
*.blend3
*.blend4
*.blend5

# Output files (don't commit generated data)
output/
simple_output/
*.hdf5

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Large files
*.hdr
*.exr
*.png
*.jpg
*.jpeg
!assets/examples/*.png
EOF

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: BlenderProc Synthetic Data Generation Pipeline

Features:
- Full BlenderProc pipeline for shackle wear detection
- Domain randomization (camera, lighting, materials)
- COCO and YOLO annotation support
- Post-processing for realistic edge noise
- Configurable via YAML

Files:
- main_pipeline.py: Full featured BlenderProc script
- simple_pipeline.py: Simplified version for quick testing
- postprocess.py: Edge noise and image augmentation
- convert_to_yolo.py: COCO to YOLO converter
- config.yaml: Pipeline configuration
- README.md: Full documentation
- QUICKSTART.md: 5-minute setup guide
"

# Set main branch
git branch -M main

# Add remote origin
git remote add origin https://github.com/Cardano-max/BlenderProc_Regression_SyntheticDataGeneration.git

echo ""
echo "============================================"
echo "  Repository initialized!"
echo "============================================"
echo ""
echo "To push to GitHub, run:"
echo "  git push -u origin main"
echo ""
echo "If you need to authenticate, you can use:"
echo "  - GitHub CLI: gh auth login"
echo "  - Personal Access Token"
echo "  - SSH key"
echo ""

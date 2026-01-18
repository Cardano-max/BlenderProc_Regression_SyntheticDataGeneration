#!/bin/bash
# ============================================================================
# BlenderProc Shackle Data Generation - Run Script (Linux/macOS)
# ============================================================================

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  BlenderProc Shackle Data Generation${NC}"
echo -e "${BLUE}============================================${NC}"

# Configuration
NUM_IMAGES=${1:-100}  # Default 100 images, or use first argument
OUTPUT_DIR=${2:-"./output"}  # Default output directory
CONFIG_FILE=${3:-"config.yaml"}  # Default config file

# Check if BlenderProc is installed
if ! command -v blenderproc &> /dev/null; then
    echo -e "${RED}Error: BlenderProc not found!${NC}"
    echo "Please install BlenderProc: pip install blenderproc"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${YELLOW}Warning: Config file not found, using defaults${NC}"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}Configuration:${NC}"
echo "  - Number of images: $NUM_IMAGES"
echo "  - Output directory: $OUTPUT_DIR"
echo "  - Config file: $CONFIG_FILE"
echo ""

# Check for assets directory
if [ ! -d "./assets/models" ]; then
    echo -e "${YELLOW}Warning: assets/models directory not found${NC}"
    echo "Please create the directory and add your .blend files"
fi

# Start time
START_TIME=$(date +%s)

echo -e "${GREEN}Starting generation...${NC}"
echo ""

# Run BlenderProc
blenderproc run main_pipeline.py \
    --config "$CONFIG_FILE" \
    --num-images "$NUM_IMAGES" \
    --output-dir "$OUTPUT_DIR" \
    --models-dir "./assets/models" \
    --hdri-dir "./assets/hdri" \
    --backgrounds-dir "./assets/backgrounds"

# Check exit status
if [ $? -eq 0 ]; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    MINUTES=$((DURATION / 60))
    SECONDS=$((DURATION % 60))
    
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}  Generation Complete!${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo "  Duration: ${MINUTES}m ${SECONDS}s"
    echo "  Output: $OUTPUT_DIR"
    echo ""
    
    # Count generated files
    if [ -d "$OUTPUT_DIR/images" ]; then
        NUM_GENERATED=$(ls -1 "$OUTPUT_DIR/images"/*.png 2>/dev/null | wc -l)
        echo "  Images generated: $NUM_GENERATED"
    fi
    
    # Run post-processing if available
    if [ -f "postprocess.py" ]; then
        echo ""
        echo -e "${YELLOW}Running post-processing...${NC}"
        python postprocess.py --input "$OUTPUT_DIR/images" --noise-level 0.02
    fi
    
else
    echo ""
    echo -e "${RED}============================================${NC}"
    echo -e "${RED}  Generation Failed!${NC}"
    echo -e "${RED}============================================${NC}"
    echo "Check the error messages above"
    exit 1
fi

echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Review generated images in $OUTPUT_DIR/images/"
echo "  2. Check annotations in $OUTPUT_DIR/annotations/"
echo "  3. Convert to YOLO format: python convert_to_yolo.py"
echo ""

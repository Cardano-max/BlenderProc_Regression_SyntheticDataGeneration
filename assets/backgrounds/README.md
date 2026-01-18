# 📁 Backgrounds Directory

Place real background image crops here for domain randomization.

## Purpose:
These are cropped sections from real photos that will be used as 
blurred backgrounds behind the shackle models. This helps the AI 
learn to ignore background variations.

## Professor's Guidance:
> "Use crops from real images... have those crops on the drive... 
> randomly choose the background to blend with the CAD model"

## How to Create Background Crops:
1. Take photos at your industrial site
2. Crop sections that show typical backgrounds (sky, trees, buildings)
3. Resize to ~1920x1080 or similar
4. Apply Gaussian blur (radius 15-30px) for consistency
5. Save as JPEG (quality 85-95%)

## Naming Convention:
- `bg_001.jpg`
- `bg_002.jpg`
- `bg_industrial_01.jpg`
- `bg_outdoor_cloudy_01.jpg`

## Supported Formats:
- `.jpg` / `.jpeg` (recommended for photos)
- `.png` (if transparency needed)

## Notes:
- More variety = better model generalization
- Include different weather conditions
- Include different times of day
- 50-100 backgrounds recommended for good variety

@echo off
REM ============================================================================
REM BlenderProc Shackle Data Generation - Run Script (Windows)
REM ============================================================================

echo ============================================
echo   BlenderProc Shackle Data Generation
echo ============================================

REM Configuration
set NUM_IMAGES=%1
if "%NUM_IMAGES%"=="" set NUM_IMAGES=100

set OUTPUT_DIR=%2
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=.\output

set CONFIG_FILE=%3
if "%CONFIG_FILE%"=="" set CONFIG_FILE=config.yaml

REM Check if BlenderProc is installed
where blenderproc >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: BlenderProc not found!
    echo Please install BlenderProc: pip install blenderproc
    exit /b 1
)

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo.
echo Configuration:
echo   - Number of images: %NUM_IMAGES%
echo   - Output directory: %OUTPUT_DIR%
echo   - Config file: %CONFIG_FILE%
echo.

echo Starting generation...
echo.

REM Run BlenderProc
blenderproc run main_pipeline.py ^
    --config "%CONFIG_FILE%" ^
    --num-images %NUM_IMAGES% ^
    --output-dir "%OUTPUT_DIR%" ^
    --models-dir ".\assets\models" ^
    --hdri-dir ".\assets\hdri" ^
    --backgrounds-dir ".\assets\backgrounds"

if %ERRORLEVEL% equ 0 (
    echo.
    echo ============================================
    echo   Generation Complete!
    echo ============================================
    echo   Output: %OUTPUT_DIR%
    echo.
    
    REM Run post-processing
    if exist "postprocess.py" (
        echo Running post-processing...
        python postprocess.py --input "%OUTPUT_DIR%\images" --noise-level 0.02
    )
    
) else (
    echo.
    echo ============================================
    echo   Generation Failed!
    echo ============================================
    echo Check the error messages above
    exit /b 1
)

echo.
echo Next steps:
echo   1. Review generated images in %OUTPUT_DIR%\images\
echo   2. Check annotations in %OUTPUT_DIR%\annotations\
echo   3. Convert to YOLO format: python convert_to_yolo.py
echo.

pause

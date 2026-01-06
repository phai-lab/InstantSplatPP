
instantsplat_inference.sh
#!/bin/bash

################################################################################
# InstantSplat Video Inference Script
# 
# Configuration:
#   - Input images: 12 views
#   - Training iterations: 3000 (best quality)
#   - GPU: CUDA device 6
#   - Output resolution: 1x (full resolution)
#
# Pipeline stages:
#   1. Geometry initialization (DUSt3R + pose estimation)
#   2. Gaussian optimization (3D-GS training with pose refinement)
#   3. Video rendering (smooth camera trajectory interpolation)
################################################################################

set -e  # Exit on error

# ============================================================================
# CONFIGURATION VARIABLES
# ============================================================================

# Paths
DATA_ROOT_DIR="your_data_root_directory"  # Set your data root directory here
OUTPUT_DIR="output_infer_mapanything_12views_3000iters"
MODEL_PATH="${OUTPUT_DIR}/model_12views_3000iters"

# Inference parameters
N_VIEWS=12                          # Number of input views
TRAIN_ITERATIONS=3000               # Training iterations (higher = better quality)
RESOLUTION=1                        # Output resolution multiplier (1 = full)
GPU_ID=5                            # CUDA device ID

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Print section header
print_header() {
    echo ""
    echo "=============================================================================="
    echo "$1"
    echo "=============================================================================="
    echo ""
}

# Print step with timestamp
print_step() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ▶ $1"
}

# Print completion message
print_complete() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ $1"
}

# ============================================================================
# MAIN INFERENCE PIPELINE
# ============================================================================

main() {
    print_header "InstantSplat Video Inference Pipeline"
    
    echo "Configuration:"
    echo "  Input Directory:     ${DATA_ROOT_DIR}"
    echo "  Output Directory:    ${MODEL_PATH}"
    echo "  Number of Views:     ${N_VIEWS}"
    echo "  Training Iterations: ${TRAIN_ITERATIONS}"
    echo "  Resolution Scale:    ${RESOLUTION}x"
    echo "  GPU Device:          ${GPU_ID}"
    echo ""
    
    # Create output directory
    mkdir -p "${MODEL_PATH}"
    
    # ========================================================================
    # STAGE 1: Geometry Initialization
    # ========================================================================
    print_header "STAGE 1: Co-visible Global Geometry Initialization"
    print_step "Initializing 3D geometry from sparse views using specific prior model..."
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore ./init_geo.py \
        -s "${SOURCE_PATH}" \
        -m "${MODEL_PATH}" \
        --n_views ${N_VIEWS} \
        --focal_avg \
        --co_vis_dsp \
        --conf_aware_ranking \
        --infer_video \
        --model_type "mapanything" \
        > "${MODEL_PATH}/01_init_geo.log" 2>&1
    
    if [ $? -eq 0 ]; then
        print_complete "Geometry initialization completed (log: 01_init_geo.log)"
    else
        echo "ERROR: Geometry initialization failed. Check ${MODEL_PATH}/01_init_geo.log"
        exit 1
    fi
    
    # ========================================================================
    # STAGE 2: Gaussian Optimization & Training
    # ========================================================================
    print_header "STAGE 2: 3D Gaussian Splatting Training"
    print_step "Training Gaussians with pose optimization (${TRAIN_ITERATIONS} iterations)..."
    print_step "This may take 2-5 minutes depending on GPU speed..."
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train.py \
        -s "${SOURCE_PATH}" \
        -m "${MODEL_PATH}" \
        -r ${RESOLUTION} \
        --n_views ${N_VIEWS} \
        --iterations ${TRAIN_ITERATIONS} \
        --pp_optimizer \
        --optim_pose \
        > "${MODEL_PATH}/02_train.log" 2>&1
    
    if [ $? -eq 0 ]; then
        print_complete "Training completed (log: 02_train.log)"
    else
        echo "ERROR: Training failed. Check ${MODEL_PATH}/02_train.log"
        exit 1
    fi
    
    # ========================================================================
    # STAGE 3: Video Rendering
    # ========================================================================
    print_header "STAGE 3: Video Rendering"
    print_step "Rendering smooth video with interpolated camera trajectory..."
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ./render.py \
        -s "${SOURCE_PATH}" \
        -m "${MODEL_PATH}" \
        -r ${RESOLUTION} \
        --n_views ${N_VIEWS} \
        --iterations ${TRAIN_ITERATIONS} \
        --infer_video \
        > "${MODEL_PATH}/03_render.log" 2>&1
    
    if [ $? -eq 0 ]; then
        print_complete "Video rendering completed (log: 03_render.log)"
    else
        echo "ERROR: Video rendering failed. Check ${MODEL_PATH}/03_render.log"
        exit 1
    fi
    
    # ========================================================================
    # Summary
    # ========================================================================
    print_header "Inference Pipeline Complete"
    
    echo "Output files saved to:"
    echo "  📁 ${MODEL_PATH}"
    echo ""
    echo "Generated files:"
    echo "  • Rendered video: renders/"
    echo "  • Point cloud: point_cloud/iteration_${TRAIN_ITERATIONS}/point_cloud.ply"
    echo "  • Training logs: 01_init_geo.log, 02_train.log, 03_render.log"
    echo ""
    echo "Pipeline execution finished successfully! ✓"
    echo ""
}

# ============================================================================
# ERROR HANDLING & EXECUTION
# ============================================================================

# Trap errors
trap 'echo "ERROR: Script interrupted or failed at line $LINENO"; exit 1' ERR

# Check if InstantSplat directory exists
if [ ! -f "./init_geo.py" ] || [ ! -f "./train.py" ] || [ ! -f "./render.py" ]; then
    echo "ERROR: InstantSplat scripts not found in current directory."
    echo "Please run this script from the InstantSplat root directory."
    exit 1
fi

# Check if input directory exists
if [ ! -d "${DATA_ROOT_DIR}" ]; then
    echo "ERROR: Input directory not found: ${DATA_ROOT_DIR}"
    exit 1
fi

# Fix SOURCE_PATH variable (was missing in original)
SOURCE_PATH="${DATA_ROOT_DIR}"

# Run main pipeline
main

exit 0
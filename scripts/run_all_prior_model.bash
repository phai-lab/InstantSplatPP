#!/bin/bash


DATA_ROOT_DIR="/data1/tzz/InstantSplatPP"  # Set your data root directory here


PRIOR_MODELS=(
    mast3r
    #vggt
    #mapanything
)


DATASETS=(
    Tanks
    # MVimgNet
)

SCENES=(
    Family
    #Horse
    #Ballroom
    #Barn
    #Church
    #Francis
    #Ignatius
    #Museum

    # bench
    # bicycle
    # car
    # chair
    # ladder
    # suv
    # table
)


N_VIEWS=(
    3
    #6
    #12
)


gs_train_iter=(
    # 200
    1000
)


get_available_gpu() {
    local mem_threshold=500
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
    $2 < threshold { print $1; exit }
    '
}


run_on_gpu() {
    local GPU_ID=$1
    local DATASET=$2
    local SCENE=$3
    local N_VIEW=$4
    local gs_train_iter=$5

    SOURCE_PATH=${DATA_ROOT_DIR}/${DATASET}/${SCENE}/24_views/
    GT_POSE_PATH=${DATA_ROOT_DIR}/${DATASET}/${SCENE}/
    IMAGE_PATH=${SOURCE_PATH}images


    MODEL_PATH=./${OUTPUT_DIR}/${DATASET}/${SCENE}/${N_VIEW}_views

    local SCENE_START
    local INIT_START
    local INIT_END
    local TRAIN_START
    local TRAIN_END
    local INIT_DURATION=0
    local TRAIN_DURATION=0

    # Create necessary directories
    mkdir -p ${MODEL_PATH}
    SCENE_START=$(date +%s)

    echo "======================================================="
    echo "Starting process: prior=${PRIOR_MODEL_TYPE} ${DATASET}/${SCENE} (${N_VIEW} views/${gs_train_iter} iters) on GPU ${GPU_ID}"
    echo "MODEL_PATH=${MODEL_PATH}"
    echo "======================================================="

    # (1) Co-visible Global Geometry Initialization
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Co-visible Global Geometry Initialization..."
    INIT_START=$(date +%s)


    CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore ./init_geo.py \
        -s ${SOURCE_PATH} \
        -m ${MODEL_PATH} \
        --n_views ${N_VIEW} \
        --focal_avg \
        --co_vis_dsp \
        --conf_aware_ranking \
        --model_type ${PRIOR_MODEL_TYPE} \
        > ${MODEL_PATH}/01_init_geo.log 2>&1
        # --ckpt_path ${PRIOR_CKPT_PATH} 

    INIT_END=$(date +%s)
    INIT_DURATION=$((INIT_END - INIT_START))
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Co-visible Global Geometry Initialization completed. Log saved in ${MODEL_PATH}/01_init_geo.log"
 
    # (2) Train: jointly optimize pose
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting training..."
    TRAIN_START=$(date +%s)
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train.py \
        -s ${SOURCE_PATH} \
        -m ${MODEL_PATH} \
        -r 1 \
        --n_views ${N_VIEW} \
        --iterations ${gs_train_iter} \
        --pp_optimizer \
        --optim_pose \
        > ${MODEL_PATH}/02_train.log 2>&1
    TRAIN_END=$(date +%s)
    TRAIN_DURATION=$((TRAIN_END - TRAIN_START))
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed. Log saved in ${MODEL_PATH}/02_train.log"
    
    # (3) Init Test Pose 
    # echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Init Test Pose..."
    # CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore ./init_test_pose.py \
    # -s ${SOURCE_PATH} \
    # -m ${MODEL_PATH} \
    # --n_views ${N_VIEW} \
    # --focal_avg \
    # > ${MODEL_PATH}/03_init_test_pose.log 2>&1
    # echo "[$(date '+%Y-%m-%d %H:%M:%S')] Init Test Pose completed. Log saved in ${MODEL_PATH}/03_init_test_pose.log"
    
    # (3) Render-Training_View
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting rendering training views..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ./render.py \
        -s ${SOURCE_PATH} \
        -m ${MODEL_PATH} \
        -r 1 \
        --n_views ${N_VIEW} \
        --iterations ${gs_train_iter} \
        > ${MODEL_PATH}/03_render_train.log 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Rendering completed. Log saved in ${MODEL_PATH}/03_render_train.log"

    # (4) Render-Testing_View
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting rendering testing views..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ./render.py \
        -s ${SOURCE_PATH} \
        -m ${MODEL_PATH} \
        -r 1 \
        --n_views ${N_VIEW} \
        --iterations ${gs_train_iter} \
        --eval \
        > ${MODEL_PATH}/04_render_test.log 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Rendering completed. Log saved in ${MODEL_PATH}/04_render_test.log"
    # --test_fps \

    # (5) Metrics
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Calculating metrics..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ./metrics.py \
        -s ${SOURCE_PATH} \
        -m ${MODEL_PATH} \
        --n_views ${N_VIEW} \
        > ${MODEL_PATH}/05_metrics.log 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Metrics calculation completed. Log saved in ${MODEL_PATH}/05_metrics.log"

    echo "======================================================="
    echo "Task completed: prior=${PRIOR_MODEL_TYPE} ${DATASET}/${SCENE} (${N_VIEW} views/${gs_train_iter} iters) on GPU ${GPU_ID}"
    echo "======================================================="

    local SCENE_END
    local TOTAL_DURATION
    SCENE_END=$(date +%s)
    TOTAL_DURATION=$((SCENE_END - SCENE_START))
    {
        printf "%s,%s,%s,%s,%s,%s,%s\n" \
        "${DATASET}" "${SCENE}" "${N_VIEW}" "${gs_train_iter}" \
        "${INIT_DURATION}" "${TRAIN_DURATION}" "${TOTAL_DURATION}"
    } >> "${TIMING_FILE}"
}



for PRIOR_MODEL_TYPE in "${PRIOR_MODELS[@]}"; do
    OUTPUT_DIR="output_eval_XL_${PRIOR_MODEL_TYPE}_multi"
    TIMING_FILE="./${OUTPUT_DIR}/scene_timings.csv"

    total_tasks=$((${#DATASETS[@]} * ${#SCENES[@]} * ${#N_VIEWS[@]} * ${#gs_train_iter[@]}))
    current_task=0
    mkdir -p "${OUTPUT_DIR}"
    echo "dataset,scene,n_views,iters,init_seconds,train_seconds,total_seconds" > "${TIMING_FILE}"

    echo "======================================================="
    echo ">>> Running PRIOR_MODEL_TYPE=${PRIOR_MODEL_TYPE}"
    echo "   OUTPUT_DIR=${OUTPUT_DIR}"
    echo "======================================================="

    for DATASET in "${DATASETS[@]}"; do
        for SCENE in "${SCENES[@]}"; do
            for N_VIEW in "${N_VIEWS[@]}"; do
                for gs_train_iter in "${gs_train_iter[@]}"; do
                    current_task=$((current_task + 1))
                    echo "Processing task $current_task / $total_tasks (prior=${PRIOR_MODEL_TYPE})"

                    # Get available GPU
                    GPU_ID=$(get_available_gpu)

                    # If no GPU is available, wait for a while and retry
                    while [ -z "$GPU_ID" ]; do
                        echo "[$(date '+%Y-%m-%d %H:%M:%S')] No GPU available, waiting 60 seconds before retrying..."
                        sleep 60
                        GPU_ID=$(get_available_gpu)
                    done

                    run_on_gpu $GPU_ID "$DATASET" "$SCENE" "$N_VIEW" "$gs_train_iter"
                done
            done
        done
    done

    echo "======================================================="
    echo "All tasks for prior=${PRIOR_MODEL_TYPE} completed! Processed $total_tasks tasks in total."
    echo "Timing CSV: ${TIMING_FILE}"
    echo "======================================================="

    if [ -f "${TIMING_FILE}" ]; then
        echo "Per-scene processing times (seconds) for prior=${PRIOR_MODEL_TYPE}:"
        awk -F',' 'NR==1 {next} {
            printf " - %s/%s (%s views, %s iters): init=%ss, train=%ss, total=%ss\n", $1, $2, $3, $4, $5, $6, $7;
            init_sum += $5; train_sum += $6; total_sum += $7; count += 1
        }
        END {
            if (count > 0) {
                printf "Average durations over %d scenes: init=%.2fs, train=%.2fs, total=%.2fs\n", count, init_sum/count, train_sum/count, total_sum/count
            }
        }' "${TIMING_FILE}"
    fi

done

echo "======================================================="
echo "All priors finished: ${PRIOR_MODELS[*]}"
echo "======================================================="

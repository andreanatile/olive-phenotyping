#!/bin/bash

# =================================================================
# Configuration: Update this variable for your project
# =================================================================

PROJECT_DIR="/home/laboratorio/repos/olive-phenotyping" 

# =================================================================
# Execution Logic (Runs Two Separate Training Jobs)
# =================================================================

echo "Starting project execution..."
echo "Navigating to: $PROJECT_DIR"

# Change directory to the project folder
cd "$PROJECT_DIR"

# Check if the directory change was successful
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Could not change directory to $PROJECT_DIR. Please check the path and permissions."
    exit 1
fi

# =================================================================
# RESIZED 640x480 TRAINING JOBS
# =================================================================
echo ""
echo "----- RUNNING RESIZED 640x480 TRAINING -----"
# --- JOB 1: Patience 10 ---
echo "Training yolo11n model for images resize to 640x480 with patience of 10 (Epochs: 300)"

EXECUTION_COMMAND_1="python3 train.py yolo11 \
/mnt/c/Datasets/OlivePG/config_70_640.yaml \
--model YOLOV11-N \
--yaml-path \"/mnt/c/Datasets/OlivePG/bbox_gt_ul_70_640/bbox_gt_ul_70_640.yaml\" \
--epochs 300 \
--patience 10"

# Execute the command
eval "$EXECUTION_COMMAND_1"

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Job 1 execution failed."
fi
echo "-----------------------------------------------------"


# =================================================================
# KEEP EMPTY PATCH TRAINING JOBS
# =================================================================
echo ""
echo "----- RUNNING RESIZED 640x480 TRAINING -----"
# --- JOB 1: Patience 10 ---
echo "Training yolo11n model for patches of 640x640 with patience of 10 (Epochs: 300)"

EXECUTION_COMMAND_2="python3 train.py yolo11 \
/mnt/c/Datasets/OlivePG/config_70_kesnb.yaml \
--model YOLOV11-N \
--yaml-path \"/mnt/c/Datasets/OlivePG/bbox_gt_ul_70_patch_kesnb/bbox_gt_ul_70_patch_kesnb.yaml\" \
--epochs 300 \
--patience 10"

# Execute the command
eval "$EXECUTION_COMMAND_2"

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Job 2 execution failed."
fi
echo "-----------------------------------------------------"

# =================================================================
# NOT KEEP EMPTY PATCH TRAINING JOBS
# =================================================================
echo ""
echo "----- RUNNING NOT KEEP EMPTY PATCH TRAINING -----"
# --- JOB 1: Patience 10 ---
echo "Training yolo11n model for patches of 640x640 with patience of 10 (Epochs: 300)"

EXECUTION_COMMAND_3="python3 train.py yolo11 \
/mnt/c/Datasets/OlivePG/config_70_nkeep.yaml \
--model YOLOV11-N \
--yaml-path \"/mnt/c/Datasets/OlivePG/bbox_gt_ul_70_patch_nkeep/bbox_gt_ul_70_patch_nkeep.yaml\" \
--epochs 300 \
--patience 10"

# Execute the command
eval "$EXECUTION_COMMAND_3"

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Job 3   execution failed."
fi
echo "-----------------------------------------------------"

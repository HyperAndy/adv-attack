#!/bin/bash
#
# run.sh is the entry point of the submission.
# nvidia-docker run -v ${INPUT_DIR}:/input_images -v ${OUTPUT_DIR}:/output_images
#       -w /competition ${DOCKER_IMAGE_NAME} sh ./run.sh /input_images /output_images
# where:
#   INPUT_DIR - directory with input png images
#   OUTPUT_DIR - directory with output png images
#

INPUT_DIR=$1
OUTPUT_DIR=$2

# python crop.py "${1}" "${2}"
CUDA_VISIBLE_DEVICES="0" python alibaba_attack_ti_ensemble_cam.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --checkpoint_path=./models/inception_v1/inception_v1.ckpt \
  --max_epsilon=32.0 \
  --num_iter=40 \
  --momentum=1.0 \
  --use_ti=False \
  --use_cross_avg=False \
  --rand_logits=False \
  --image_resize=170

 CUDA_VISIBLE_DEVICES="0" python grad_cam1.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}"
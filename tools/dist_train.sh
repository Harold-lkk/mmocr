#!/usr/bin/env bash

set -x
set -e

CONFIG=$1
WORK_DIR=$2
GPUS=$3

PORT=${PORT:-29500}

rm -rf ~/.parrots

python $(dirname "$0")/train.py $CONFIG --work-dir=${WORK_DIR} --launcher mpi ${@:4}
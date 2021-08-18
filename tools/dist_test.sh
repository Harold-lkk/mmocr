#!/usr/bin/env bash

set -x
set -e

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29500}

rm -rf ~/.parrots

python $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher mpi ${@:4}

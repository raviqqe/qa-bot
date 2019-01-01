#!/bin/sh

set -ex

dataset_file=$1
shift

./create_vocab.py $dataset_file vocab.json
./train.py --vocab vocab.json --dataset $dataset_file --model model.npz "$@"

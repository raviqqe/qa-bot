#!/bin/sh

set -e

./create_vocab.py test/dataset.jl vocab.json
./train.py --vocab vocab.json --dataset test/dataset.jl --model model.npz

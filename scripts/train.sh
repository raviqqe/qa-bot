#!/bin/sh

set -e

./create_vocab.py test/dataset.txt vocab.json
./train.py --vocab vocab.json --dataset test/dataset.txt --model model.npz

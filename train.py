#!/usr/bin/env python3

import argparse
import json

import chainer
import numpy

from constants import UNKNOWN
from model import DeepThought


def load_dataset(filename, vocab):
    def transform_dataset(line):
        item = json.loads(line)

        return tuple(
            numpy.array([vocab.get(char, UNKNOWN) for char in sentence.strip()])
            for sentence in [item["question"], item["answer"]]
        )

    return chainer.datasets.TransformDataset(
        chainer.datasets.TextDataset(filename), transform_dataset
    )


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vocab", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--units", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--resume", default="")
    parser.add_argument("--save", default="")

    return parser.parse_args()


def main():
    args = get_args()

    vocab = json.load(open(args.vocab))
    dataset = load_dataset(args.dataset, vocab)

    print("samples =", len(dataset))
    print("chars =", len(vocab))

    model = DeepThought(args.layers, len(vocab), args.units)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    log_trigger = (args.log_interval, "iteration")

    trainer = chainer.training.Trainer(
        chainer.training.updaters.StandardUpdater(
            chainer.iterators.SerialIterator(dataset, args.batch), optimizer
        ),
        stop_trigger=(args.iterations, "iteration"),
    )
    trainer.extend(chainer.training.extensions.LogReport(trigger=log_trigger))
    trainer.extend(
        chainer.training.extensions.PrintReport(
            ["epoch", "iteration", "main/loss", "elapsed_time"]
        ),
        trigger=log_trigger,
    )

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

    if args.save:
        chainer.serializers.save_npz(args.save, trainer)

    model.save(args.model)


if __name__ == "__main__":
    main()

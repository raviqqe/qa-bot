#!/usr/bin/env python3

import argparse

import chainer
import numpy

from constants import N_DUMMY_CHARS
from model import DeepThought


def load_dataset(filename):
    # Read a TSV file of (question, answer).

    vocab = {
        char: index + N_DUMMY_CHARS  # skip unknown and EOS characters
        for index, char in enumerate(sorted({char for char in open(filename).read()}))
    }

    def transform_dataset(line):
        return tuple(
            numpy.array([vocab[char] for char in sentence])
            for sentence in line.split("\t")[:2]
        )

    return (
        chainer.datasets.TransformDataset(
            chainer.datasets.TextDataset(filename), transform_dataset
        ),
        vocab,
    )


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_filename")
    parser.add_argument("model_filename")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--units", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--resume", default="")
    parser.add_argument("--save", default="")

    return parser.parse_args()


def main():
    args = get_args()

    dataset, vocab = load_dataset(args.dataset_filename)

    print("samples =", len(dataset))
    print("chars =", len(vocab))

    model = DeepThought(args.layers, len(vocab) + N_DUMMY_CHARS, args.units)

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

    model.save(args.model_filename)


if __name__ == "__main__":
    main()

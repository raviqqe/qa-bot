#!/usr/bin/env python3

import argparse
import json

import chainer
import numpy

from constants import EOS, UNKNOWN
from model import DeepThought

SENTENCE_LENGTH = 256  # Twitter's limit is 280 characters.


def create_pad(array):
    return numpy.full(SENTENCE_LENGTH - len(array), EOS)


def create_sentence_array(sentence, vocab):
    return numpy.array(
        [vocab.get(char, UNKNOWN) for char in sentence[-SENTENCE_LENGTH:]]
    )


def create_question_array(question, vocab):
    array = create_sentence_array(question, vocab)
    return numpy.concatenate([create_pad(array), array])


def create_answer_array(answer, vocab):
    array = create_sentence_array(answer, vocab)
    return numpy.concatenate([array, create_pad(array)])


def load_dataset(filename, vocab):
    def transform_dataset(line):
        item = json.loads(line)

        return (
            create_question_array(item["question"], vocab),
            create_answer_array(item["answer"], vocab),
        )

    return chainer.datasets.TransformDataset(
        chainer.datasets.TextDataset(filename), transform_dataset
    )


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vocab", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--units", type=int, default=256)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0)
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

    model = DeepThought(
        n_layers=args.layers,
        n_chars=len(vocab),
        n_units=args.units,
        dropout=args.dropout,
    )

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

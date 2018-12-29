#!/usr/bin/env python3

import argparse

import chainer
import chainer.functions as F
import chainer.links as L
import numpy

EOS = 0
N_DUMMY_CHARS = 1


class DeepThought(chainer.Chain):
    def __init__(self, n_layers, n_chars, n_units):
        super(DeepThought, self).__init__()

        with self.init_scope():
            self.embed = L.EmbedID(n_chars, n_units)
            self.encoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
            self.decoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
            self.W = L.Linear(n_units, n_chars)

    def forward(self, xs, ys):
        xs = [x[::-1] for x in xs]

        hs, cs, _ = self.encoder(None, None, self.sequence_embed(self.embed, xs))
        _, _, zs = self.decoder(
            hs,
            cs,
            self.sequence_embed(
                self.embed, [F.concat([numpy.array([EOS]), y], axis=0) for y in ys]
            ),
        )

        loss = F.sum(
            F.softmax_cross_entropy(
                self.W(F.concat(zs, axis=0)),
                F.concat(
                    [F.concat([y, numpy.array([EOS])], axis=0) for y in ys], axis=0
                ),
                reduce="no",
            )
        ) / len(xs)

        chainer.report({"loss": loss}, self)

        return loss

    @staticmethod
    def sequence_embed(embed, xs):
        return F.split_axis(
            embed(F.concat(xs, axis=0)), numpy.cumsum([len(x) for x in xs[:-1]]), 0
        )


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

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(DeepThought(args.layers, len(vocab) + N_DUMMY_CHARS, args.units))

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


if __name__ == "__main__":
    main()

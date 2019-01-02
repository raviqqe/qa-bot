import chainer
import chainer.functions as F
import chainer.links as L
import numpy

from constants import EOS


class DeepThought(chainer.Chain):
    def __init__(self, *, n_layers, n_chars, n_units, dropout=0):
        super(DeepThought, self).__init__()

        with self.init_scope():
            self.embed = L.EmbedID(n_chars, n_units)
            self.encoder = L.NStepLSTM(n_layers, n_units, n_units, dropout)
            self.decoder = L.NStepLSTM(n_layers, n_units, n_units, dropout)
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

    def answer(self, xs, max_length=100):
        with chainer.no_backprop_mode(), chainer.using_config("train", False):
            ys = []
            hs, cs, _ = self.encoder(
                None, None, self.sequence_embed(self.embed, [x[::-1] for x in xs])
            )
            y = numpy.full(len(xs), EOS)

            for _ in range(max_length):
                hs, cs, os = self.decoder(
                    hs, cs, F.split_axis(self.embed(y), len(xs), 0)
                )
                y = numpy.argmax(self.W(F.concat(os, axis=0)).array, axis=1)
                ys.append(y)

        return [
            y[: numpy.argwhere(y == EOS)[0, 0]]
            for y in numpy.stack([*ys, numpy.full(len(xs), EOS)]).T
        ]

    def save(self, path):
        chainer.serializers.save_npz(path, self)

    def load(self, path):
        chainer.serializers.load_npz(path, self)

    @staticmethod
    def sequence_embed(embed, xs):
        return F.split_axis(
            embed(F.concat(xs, axis=0)), numpy.cumsum([len(x) for x in xs[:-1]]), 0
        )

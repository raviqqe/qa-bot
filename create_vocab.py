#!/usr/bin/env python3

import argparse
import json

from constants import EOS, N_DUMMY_CHARS, UNKNOWN


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_filename")
    parser.add_argument("vocab_filename")

    return parser.parse_args()


def main():
    args = get_args()

    json.dump(
        {
            "": EOS,
            "\u200b": UNKNOWN,
            **{
                char: index + N_DUMMY_CHARS  # skip unknown and EOS characters
                for index, char in enumerate(
                    sorted({char for char in open(args.dataset_filename).read()})
                )
            },
        },
        open(args.vocab_filename, "w"),
    )


if __name__ == "__main__":
    main()

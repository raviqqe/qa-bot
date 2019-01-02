#!/usr/bin/env python3

import json
import os
import threading
import time

import flask
import numpy
import twitter

from constants import UNKNOWN
from model import DeepThought


class Bot(threading.Thread):
    def __init__(self):
        super().__init__()

        self.vocab = json.load(open(os.environ["VOCAB_FILE"]))
        self.reverse_vocab = {index: char for char, index in self.vocab.items()}

        self.model = DeepThought(
            n_layers=int(os.environ["N_LAYERS"]),
            n_chars=len(self.vocab),
            n_units=int(os.environ["N_UNITS"]),
        )
        self.model.load(os.environ["MODEL_FILE"])

        self.replied_status_ids = set()
        self.api = twitter.Api(
            consumer_key=os.environ["CONSUMER_KEY"],
            consumer_secret=os.environ["CONSUMER_SECRET"],
            access_token_key=os.environ["ACCESS_TOKEN_KEY"],
            access_token_secret=os.environ["ACCESS_TOKEN_SECRET"],
            sleep_on_rate_limit=True,
        )

    def run(self):
        while True:
            for status in self.api.GetMentions(count=1):
                if status.id in self.replied_status_ids:
                    continue

                self.replied_status_ids.add(status.id)
                self.reply(status)

            time.sleep(20)

    def reply(self, status):
        try:
            print("Question:", status.text)

            answer = "".join(
                self.reverse_vocab.get(index, "💩")
                for index in self.model.answer(
                    numpy.array(
                        [[self.vocab.get(char, UNKNOWN) for char in status.text]]
                    )
                )[0]
            )

            print("Answer:", answer)

            self.api.PostUpdate(
                answer,
                in_reply_to_status_id=status.id,
                auto_populate_reply_metadata=True,
            )
        except twitter.error.TwitterError as error:
            print("TwitterAPIError:", error)


Bot().start()

app = flask.Flask(__name__)


@app.route("/")
def health_check():
    return "Bot is healthy!"

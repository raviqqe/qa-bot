#!/usr/bin/env python3

import datetime
import flask
import os
import threading
import time
import twitter


def log(kind, *data):
    date = datetime.datetime.utcnow()
    print(f"[{date}] {kind}:", *data)


class Bot(threading.Thread):
    def __init__(self):
        super().__init__()

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
            log("Question", status.text)

            answer = "Hi! This is a QA bot!"

            log("Answer", answer)

            self.api.PostUpdate(
                answer,
                in_reply_to_status_id=status.id,
                auto_populate_reply_metadata=True,
            )
        except twitter.error.TwitterError as error:
            log("TwitterAPIError:", error)


Bot().start()

app = flask.Flask(__name__)


@app.route("/")
def health_check():
    return "Bot is healthy!"

import re

import scrapy


class ChiebukuroSpider(scrapy.Spider):
    name = "chiebukuro"
    allowed_domains = ["chiebukuro.yahoo.co.jp"]
    start_urls = ["http://chiebukuro.yahoo.co.jp/"]

    def parse(self, response):
        for url in response.css("a::attr('href')").extract():
            if not url.startswith("https://"):
                continue
            elif "https://detail.chiebukuro.yahoo.co.jp/qa/question_detail/" in url:
                yield scrapy.Request(url, callback=self.parse_item)
            yield scrapy.Request(url, callback=self.parse)

    def parse_item(self, response):
        question = self.clean_text(
            "".join(response.css("div.sttsRslvd div.ptsQes p::text").extract())
        )

        if not question:
            return

        answer = self.clean_text(
            "".join(response.css("div.mdPstdBA div.ptsQes p::text").extract())
        )

        if not answer:
            return

        yield {"question": question, "answer": answer}

    @staticmethod
    def clean_text(text):
        return re.sub(r"\s+", " ", text).strip()

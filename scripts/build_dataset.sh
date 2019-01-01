#!/bin/sh

set -e

cd dataset
scrapy crawl chiebukuro --logfile dataset.log -o dataset.jl

#!/bin/sh

set -e

for var in CONSUMER_KEY CONSUMER_SECRET ACCESS_TOKEN_KEY ACCESS_TOKEN_SECRET
do
	echo "  $var: $(eval "echo \$$var")" >> app.yaml
done

gcloud app deploy --version production

git checkout app.yaml

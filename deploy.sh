#!/bin/sh

echo "env_variables:" >> app.yaml

for var in CONSUMER_KEY CONSUMER_SECRET ACCESS_TOKEN_KEY ACCESS_TOKEN_SECRET
do
	echo "  $var: $(eval "echo \$$var")" >> app.yaml
done

gcloud app deploy

git checkout app.yaml

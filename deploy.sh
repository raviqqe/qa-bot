#!/bin/sh

echo "env_variables:" >> app.yaml
echo "  CONSUMER_KEY: $CONSUMER_KEY" >> app.yaml
echo "  CONSUMER_SECRET: $CONSUMER_SECRET" >> app.yaml
echo "  ACCESS_TOKEN_KEY: $ACCESS_TOKEN_KEY" >> app.yaml
echo "  ACCESS_TOKEN_SECRET: $ACCESS_TOKEN_SECRET" >> app.yaml

gcloud app deploy

git checkout .

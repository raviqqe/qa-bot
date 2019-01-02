FROM python:3

COPY . /src

WORKDIR /src

RUN apt -y update --fix-missing && apt -y install jq
RUN pip3 install --user -r requirements.txt

RUN pip3 install black
RUN black --check .

RUN scripts/train.sh test/dataset.jl
RUN jq '.[-1]["main/loss"]' result/log | cut -f 1 -d . | grep '^[0-3]$'

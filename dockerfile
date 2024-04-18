FROM docker.io/library/ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        build-essential \
        python3-pip \
        python3-dev \
        wget \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy necessary scripts
RUN mkdir -p geogiant
RUN mkdir -p geogiant/datasets

COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock

COPY geogiant/clickhouse geogiant/clickhouse
COPY geogiant/common geogiant/common
COPY geogiant/ecs_vp_selection/scores.py geogiant/
COPY geogiant/score_datasets/targets_subnet.json geogiant/datasets
COPY geogiant/score_datasets/vps_subnet.json geogiant/datasets
COPY README.md README.md

# install dependencies
RUN pip3 install --no-cache-dir poetry
RUN poetry config virtualenvs.in-project true

RUN poetry lock && \
    poetry install

ENTRYPOINT poetry run python geogiant/scores.py

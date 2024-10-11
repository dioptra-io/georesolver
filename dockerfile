FROM docker.io/library/ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        build-essential \
        python3-pip \
        python3-dev \
        wget \
        curl \
        git \
        apt-transport-https \
        ca-certificates \
        gnupg \
    && rm -rf /var/lib/apt/lists/*

# Download and install Go
RUN wget https://golang.org/dl/go1.22.2.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go1.22.2.linux-amd64.tar.gz && \
    rm go1.22.2.linux-amd64.tar.gz

# Set Go environment variables
ENV PATH="/usr/local/go/bin:${PATH}"
ENV GOPATH="/go"

    # Download and install clickhouse
RUN curl -fsSL 'https://packages.clickhouse.com/rpm/lts/repodata/repomd.xml.key' | gpg --dearmor -o /usr/share/keyrings/clickhouse-keyring.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/clickhouse-keyring.gpg] https://packages.clickhouse.com/deb stable main" | tee \
    /etc/apt/sources.list.d/clickhouse.list
RUN apt-get update && apt-get install --yes clickhouse-client

WORKDIR /app

# copy necessary scripts
RUN mkdir -p geogiant
RUN mkdir -p logs
RUN mkdir -p geogiant/experiments
RUN mkdir -p geogiant/datasets

COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock

COPY geogiant/zdns geogiant/zdns
COPY geogiant/prober geogiant/prober
COPY geogiant/common geogiant/common
COPY geogiant/agent/ geogiant/agent
COPY geogiant/clickhouse geogiant/clickhouse

COPY geogiant/ripe_init.py geogiant/ripe_init.py
COPY geogiant/ecs_geoloc_eval.py geogiant/ecs_geoloc_eval.py
COPY geogiant/ecs_mapping_init.py geogiant/ecs_mapping_init.py
COPY geogiant/main.py geogiant/main.py

COPY README.md README.md

# install zdns
RUN git clone https://github.com/zmap/zdns.git
RUN cd zdns && go build && cd -
RUN cp zdns/zdns geogiant/zdns/zdns_binary && rm -rf zdns/

# install dependencies
RUN pip3 install --no-cache-dir poetry
RUN poetry config virtualenvs.in-project true

RUN poetry lock && \
    poetry install
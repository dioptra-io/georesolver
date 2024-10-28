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

# Download and install clickhouse client
RUN curl -fsSL 'https://packages.clickhouse.com/rpm/lts/repodata/repomd.xml.key' | gpg --dearmor -o /usr/share/keyrings/clickhouse-keyring.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/clickhouse-keyring.gpg] https://packages.clickhouse.com/deb stable main" | tee \
    /etc/apt/sources.list.d/clickhouse.list
RUN apt-get update && apt-get install --yes clickhouse-client

WORKDIR /app

# copy necessary scripts
RUN mkdir -p georesolver
RUN mkdir -p logs
RUN mkdir -p georesolver/experiments
RUN mkdir -p georesolver/datasets

COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock

COPY georesolver/zdns georesolver/zdns
COPY georesolver/prober georesolver/prober
COPY georesolver/common georesolver/common
COPY georesolver/agent/ georesolver/agent
COPY georesolver/clickhouse georesolver/clickhouse

COPY georesolver/main.py georesolver/main.py

COPY README.md README.md

# install zdns
RUN git clone https://github.com/zmap/zdns.git
RUN cd zdns && go build && cd -
RUN cp zdns/zdns georesolver/zdns/zdns_binary && rm -rf zdns/

# install dependencies
RUN pip3 install --no-cache-dir poetry
RUN poetry config virtualenvs.in-project true

RUN poetry lock && \
    poetry install
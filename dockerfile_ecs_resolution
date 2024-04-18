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

# Download and install Go
RUN wget https://golang.org/dl/go1.22.2.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go1.22.2.linux-amd64.tar.gz && \
    rm go1.22.2.linux-amd64.tar.gz

# Set Go environment variables
ENV PATH="/usr/local/go/bin:${PATH}"
ENV GOPATH="/go"

WORKDIR /app

# copy necessary scripts
RUN mkdir -p geogiant
RUN mkdir -p geogiant/datasets

COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock

COPY geogiant/clickhouse geogiant/clickhouse
COPY geogiant/common geogiant/common
COPY geogiant/zdns geogiant/zdns
COPY geogiant/hostname_init.py geogiant/
COPY geogiant/hostname_datasets/vps_subnet.json geogiant/datasets
COPY geogiant/hostname_datasets/rib_table.dat geogiant/datasets
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

ENTRYPOINT poetry run python geogiant/hostname_init.py

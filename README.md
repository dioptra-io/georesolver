# GeoResolver

Internet scale IP address geolocation tool. Measurement are run on RIPE Atlas.

## Test environments

Only tested on Ubuntu 22.04 LTS and centOS 7.
No guarantees to work on other OS.
Docker X.X
Python X.X
Clickhouse X.X

## Pre-requestists

Docker
Python 3.x
Clickhouse

## Setup env variables

At the root GeoResolver project, create a .env file and setup required environement variables.
An example of necessary env variables are given in .env.example.

## Clickhouse

Install clickhouse client

Start clickhouse server docker image
docker run -d \
    -p 127.0.0.1:8123:8123 \
    -p 127.0.0.1:9000:9000 \
    -e CLICKHOUSE_MAX_QUERY_SIZE=10000000000000000000 \
    -v /srv/hugo/GeoResolver_repro/clickhouse/data:/var/lib/clickhouse/ \
    -v /srv/hugo/GeoResolver_repro/clickhouse/logs:/var/log/clickhouse-server/ \
    --ulimit nofile=262144:262144 \
    clickhouse/clickhouse-server:22.6

Replace storage path var with the path to your storage (required as replication requires a considerable amount of data to be downloaded).

Create GeoResolver database:

clickhouse-client 

## Install project

GeoResolver uses poetry as python virtual environement and packet manager, you can install it with:
install poetry:
```bash
pip install poetry
```

```bash
cd georesolver
poetry lock && poetry install
```

## Activate virtual env

```bash
poetry shell
```

## Set up .env

GeoResolver relies on .env file to load credentials. 
see env.example for an example of the necessary environment variable. 

## Download historic data

### Download tables from FTP

Download artifact tables from FTP server, password/user upon request

### Insert table into Clickhouse

Use python installer

### Test

## run local script

Once in the poetry virtual env, one can run georesolver using pre-defined scripts:

```bash
python georesolver/scripts/local_demo.py
```

Or execute GeoResolver with the docker image:

```bash
python georesolver/scripts/local_docker_demo.py
```

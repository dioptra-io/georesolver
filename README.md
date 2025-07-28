# 🗺️ GeoResolver

GeoResolver is an Internet scale IP address geolocation engine (both IPv4 and IPv6) made for running on RIPE Atlas. Its main purpose is to infer the relative distance between a target IP address and a set of vantage points with known geolocation using ECS-DNS redirection similarity. Based on this similarity GeoResolver select a fixed set of 50 vantage points from RIPE Atlas and perform pings, acheiving near optimal precision (the optimal one being when using all the vantage points). 

# 📋 Content

This repository contains all code and workflows to **replicate** and **reproduce** GeoResolver's main results. You can run your own experiments to geolocate a set of target of IP addresses of your chosing. Note that GeoResolver relies on RIPE Atlas to perform ping measurement and infer the geolocation of an IP address, therefore, you will need a RIPE Atlas account and some credits to run measurements. All part of GeoResolver are **open-source** and **publically available**.

# 📚 References

- **Title** : GeoResolver: An Accurate, Scalable, and Explainable Geolocation Technique Using DNS 
Redirection
- **Authors** : Hugo Rimlinger, Olivier Fourmaux, Timur Friedman, and Kevin Vermeulen
- CoNEXT 2025 paper (not yet available)
- ACM artifacts (not yet available)

# 🚩 Prerequisites

GeoResolver relies on a couple of dependencies:
- [Docker >= 28.0.1](https://docs.docker.com/engine/install/)
- [Python >= 3.10.12](https://www.python.org/downloads/)
- [Golang >= 1.23.1](https://go.dev/doc/install)

The following of this tutorial will guide you throught the installation step of all other ressources.
Notice that, because of the large scale of the measurement GeoResolver's can handle, artifacts are quite large. Therefore, GeoResolver's requires at least X Gb of storage to be fully replicable. A smaller version is also available for those who are only interested into running their own experiments.

# 🧪 Test environments

We tested GeoResolver on the following software/hardware setup:
- OS: Ubuntu 22.04.5 LTS
- CPU: Intel(R) Xeon(R) Gold 5122 CPU @ 3.60GHz, x86_64, 12 cores
- RAM: 250G
- Docker: 28.0.1
- Python: 3.10.12
- Poetry: 1.1.12
- Clickhouse-server: 22.6
- Clickhouse-client: 22.2.2.1
- ZDNS: 2.0.4

# 🔧 Installation

The following steps will guide you throught GeoResolver's project installation.

## Install GeoResolver

First of all, clone this repository:
```bash
git clone https://github.com/dioptra-io/georesolver.git
cd georesolver
```

GeoResolver uses poetry as python virtual environment and packet manager, you can install it with:
```bash
pip install poetry
```

Finally, install GeoResolver:
```bash
poetry lock && poetry install
```

Depending on the version of poetry you installed, you can activate GeoResolver virtual environement either with:
```bash
poetry shell
```

Or, in new poetry version, virtual env is activated with:
```bash
poetry env activate
```
And python scripts are run with:
```bash
poetry run python <my_beautiful_script.py>
```

## Install Clickhouse

GeoResolver uses [Clickhouse](https://clickhouse.com/docs) as database management system. First, install **Clickhouse client** with:
```bash
curl -fsSL 'https://packages.clickhouse.com/rpm/lts/repodata/repomd.xml.key' | gpg --dearmor -o /usr/share/keyrings/clickhouse-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/clickhouse-keyring.gpg] https://packages.clickhouse.com/deb stable main" | tee \
    /etc/apt/sources.list.d/clickhouse.list
apt-get update && apt-get install --yes clickhouse-client
```
If outadated, check [Clickhouse installation](https://clickhouse.com/docs/install) link for updated installation commands.

Once done, install and start the **Clickhouse server** using docker:
```bash
docker run -d \
    -p 127.0.0.1:8123:8123 \
    -p 127.0.0.1:9000:9000 \
    -e CLICKHOUSE_MAX_QUERY_SIZE=10000000000000000000 \
    -v <clickhouse_storage_dir>/data:/var/lib/clickhouse/ \
    -v <clickhouse_storage_dir>/logs:/var/log/clickhouse-server/ \
    --ulimit nofile=262144:262144 \
    clickhouse/clickhouse-server:22.6
```

Finally, create GeoResolver database using the following command:
```bash
clickhouse-client --query="CREATE DATABASE IF NOT EXISTS GeoResolver"
```

## Install ZDNS

GeoResolver relies on ECS-DNS requests to select the best set of vantage points to geolocate a given target IP address. We used [ZDNS](https://github.com/zmap/zdns), a high-speed DNS resolver to perform these queries.

First, check that you have correctly installed golang:
```bash
go version
```

If not, install it with:
```bash
wget https://golang.org/dl/go1.22.2.linux-amd64.tar.gz

tar -C /usr/local -xzf go1.22.2.linux-amd64.tar.gz

rm go1.22.2.linux-amd64.tar.gz
```

Once done, clone ZDNS github repository, compile the project and copy the produced binary into ZDNS's directory:
```bash
git clone https://github.com/zmap/zdns.git
cd zdns && go build && cd -
cp zdns/zdns georesolver/zdns/zdns_binary && rm -rf zdns/
```

Check that ZDNS installation works properly:
```bash
echo "www.example.org" | georesolver/zdns/zdns_binary A --name-servers 8.8.8.8
```

## Setup credentials

GeoResolver search all credentials into a .env files. First create it at the root of this repository:
```bash
touch .env
```
And copy-paste the content of **.env.example** within it. If you made changes (such as CLickhouse password/username or database), modify the .env file accordingly.

# 🗂️ Download artifacts

Because of their size, all GeoResolver artifacts are available on our FTP serveur at: [ftp://132.227.123.74/CoNEXT_artifacts/]().
These artifacts are composed of two parts:
1. A zip archive withh all necessary files, such as domain names, CAIDA AS to organization, ITDK ICMP responsive IP addresses, etc.
2. All necessary Clickhouse tables.

## Download and extract archive

First, download the archive containing all necessary artifacts with:
```bash
curl -u CoNEXT_artifacts_user:61d0b0f7-0c54-4196-886b-4bcc6b0a0638 'ftp://132.227.123.74/CoNEXT_artifacts/datasets.zip' -o datasets.zip

mkdir -p georesolver/datasets

unzip datasets.zip -d georesolver/datasets/
```

**Description**:
- itdk:
- static_files:
- hostname_files:
- experiment_configs:
- other files:

## Download and install Clickhouse tables

We provide an installer to download and install Clickhouse tables in your own local instance. Simply run:
```bash
python configuration/installer.py
```
or 
```bash
poetry run python configuration/installer.py
```

This script will download all files listed as input parameters from the FTP server, create the corresponding table into Clickhouse (DNS/Ping/etc.) and import the data. Below you will find a short description of the default tables.

**Description**
- Ping tables:
    - vps_meshed_pings:
    - meshed_ping_cdns:
- DNS tables:
    - vps_ecs_mapping:
    - meshed_ecs_cdns:
- VPs tables:
    - vps_raw:
    - vps_filtered:
    - vps_filtered_final:


Gongratulation, you are throught with GeoResolver's installation!

# 🔬 Run Evaluation

All results present in GeoResolver's original paper are fully replicable. All the necessary scripts can be found in [georesolver/evaluation/](georesolver/evaluation/).
For the sake of simplicity, we propose a single script to run all evaluation at: [georesolver/evaluation/evaluation_all.py](georesolver/evaluation/evaluation_all.py):
```bash
python georesolver/evaluation/evaluation_all.py
```
or 
```bash
poetry run python georesolver/evaluation/evaluation_all.py
```

Otherwise, you can run each script separately. You will notice that some step of these evaluation scripts are skipped. It is because these scripts can also be used to run the measurement used for the evaluation. **WE DO NOT RECOMMEND** running them as these measurement put a significant load on RIPE Atlas plateform. Indeed, most of them perform *meshed measurements*, i.e. measurement from all vantage points to a set of targets, which is very costly.

Instead, in the following section, we propose you to run your own measurements using GeoResolver, as it is leightweight and operate with a fixed cost of 50 vantage points to geolocate each target IP address.

# 🗺️ Run your own experiments

Run GeoResolver script:
```bash
python georesolver/scripts/local_demo.py
```
or
```bash
poetry run python georesolver/scripts/local_demo.py
```

Note that this section will require you to have a RIPE Atlas account with enought credits to run your measurement. Create your [RIPE Atlas account]() if you do not have one and request credits if needed (RIPE Atlas offers credits to researchers when requested). 

This python script select a random set of 100 IP addresses from the ITDK ICMP responsive IP addresses. It then run GeoResolver geolocation engine on this set of IP addresses. GeoResolver start 4 processes in parrallel:
1. **ECS process**: Run ECS-DNS queries on all target's subnets.
2. **Score process**: Compute the redirection distance between the ECS resolution of over all vantage points and the one obtain for the targets.
3. **Ping process**: Select the vantage points with the smallest redirection distance and post the measurement on RIPE Atlas.
4. **Insert process**: Insert ping measurement into a clickhouse table.

GeoResolver adopt a *waterfall* design, meaning that each process works in batch, starting with the ECS process. Once a batch is done, it triggers the Score process which, when done, triggers the Ping process and so on. You can follow the execution of each process by inspecting the logs at [georesolver/logs/local_demo/]() where each process has its own log file.

At the end, you should have 4 tables: local_demo_ecs, local_demo_score, local_demo_ping and local_demo_geoloc (which contains the result of the geolocation based on the shortest ping vantage point geolocation). Addiotionally, you can run [georesolver/scripts/geoloc_map.py](georesolver/scripts/geoloc_map.py) to obtain a map of your geolocated IP addresses (by default this script only output IP addresses geolocated under 2ms):
```bash
python geresolver/scripts/geoloc_map.py 
```

Notice that you can choose to load any target file as input (this is the script we used to run GeoResolver on the ITDK dataset), just check that your IP addresses answer to pings! (As mentionned in the paper, GeoResolver methodology works for IPv6 but we did not yet have the tool available for it).


# Acknowledgements

We thank the anonymous reviewers from the ACM CoNEXT program committee for their thoughtful reviews. We thank Maxime Mouchet from IPInfo for sharing their geolocation data via their academic program. Hugo Rimlinger, Olivier Fourmaux, and Timur Friedman are supported by a grant from the French Ministry of Armed Forces.

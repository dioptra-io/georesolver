# 🗺️ GeoResolver

GeoResolver is an internet-scale IP address geolocation engine (supporting both IPv4 and IPv6), designed to operate on the RIPE Atlas platform. Its primary objective is to infer the relative distance between a target IP address and a set of vantage points with known geolocations, leveraging ECS-DNS redirection similarity.

Based on this similarity metric, GeoResolver selects a fixed subset of 50 RIPE Atlas vantage points and performs ping measurements, achieving near-optimal geolocation accuracy — comparable to using the full set of available vantage points.

# 📋 Content

This repository contains all the code, workflows, and data required to both replicate and reproduce the main results of GeoResolver.

- **Replication** refers to rerunning our original experiments with the same inputs to validate the published results.

- **Reproduction** enables researchers to apply the methodology to new datasets or target IP addresses of their choosing.

Note that GeoResolver relies on RIPE Atlas for performing active ping measurements, which are central to its geolocation inference process. To run measurements, you will need a [RIPE Atlas account](https://atlas.ripe.net/) and sufficient credits.

All components of GeoResolver are **open-source and publicly available**.

# 📚 References

- **Title** : GeoResolver: An Accurate, Scalable, and Explainable Geolocation Technique Using DNS 
Redirection
- **Authors** : Hugo Rimlinger, Olivier Fourmaux, Timur Friedman, and Kevin Vermeulen
- CoNEXT 2025 paper (not yet available)
- ACM artifacts (not yet available)

# 🚩 Prerequisites

GeoResolver relies on the following dependencies:
- [Docker >= 28.0.1](https://docs.docker.com/engine/install/)
- [Python >= 3.10.12](https://www.python.org/downloads/)
- [Golang >= 1.23.1](https://go.dev/doc/install)

The remainder of this tutorial will guide you through the installation of all other required resources.
Note: Due to the large-scale nature of GeoResolver, the associated artifacts requires sufficient storage. To fully replicate the original experiments, we recommend having at least 15 GB of available storage (total size of artifacts is 9.4GB). 

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

GeoResolver relies on ECS-enabled DNS requests to identify the most suitable set of vantage points for geolocating a given target IP address. To perform these queries, we use [ZDNS](https://github.com/zmap/zdns), a high-performance DNS resolver capable of handling large-scale lookups at speed.

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
Copy-paste the content of **.env.example** within it. If you made changes (such as CLickhouse password/username or database), modify the .env file accordingly. Do not forget to set your RIPE Atlas API key as well.

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
- **itdk**: This folder contains all files related with the ITDK dataset (all IP addresses/ICMP responsive IP addresses, etc.). Original dataset can be found at [CAIDA ITDK](https://www.caida.org/catalog/datasets/internet-topology-data-kit/)
- **static_files**: This folder contains some useful datasets like [CAIDA AS to organization](https://www.caida.org/archive/as2org/), BGP prefixes from [Anycatch](https://bgp.tools/kb/anycatch), etc.
- **hostname_files**: Hostname files are all intermediary files we used and produced to select the hostname GeoResolver relies on to calculate the redirection distance between vantage points' subnets and targets one, starting with the [top 1M CrUX table](https://github.com/zakird/crux-top-lists).
- **experiment_configs**: Some configuration example to use GeoResolver along with its docker image.
- other files: These other files contains vantage points that were filter because of wrongful geolocation, best hostnames retrieved ranked function of their hosting diversity, etc.

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

- **Ping tables**:
    - **itdk_ping**: Contains the result of all ping measurement made with GeoResolver of the 2.1M ICMP responsive IP addresses of the ITDK dataset.
    - **vps_meshed_pings**: Contains the result of pings from all RIPE Atlas probes to all RIPE Atlas anchors. We use this dataset to compare GeoResolver vantage point selection vs. optimal precision (i.e. all vantage points) and to filter out wrongly geolocated probes (using speed of light violation).
    - **vps_meshed_pings_CoNEXT_summer_submision**: Historic vps_meshed_pings dataset ran for the winter submission of CoNEXT 2025.
    - **vps_meshed_pings_ipv6**: Same a vps_meshed_pings but for IPv6.
    - **meshed_ping_cdns**: Same as vps_meshed_pings but this time towards one IP address in every /24 prefix we retrieved when performing ECS-DNS resolution of all /24 vantage points' prefixes. We use this dataset to evaluate the optimality of the redirection, in respect to the observed latency.
    - **single_radius_ping**: Contains the result of measurement made by RIPE Atlas geolocation engine [single-radius](https://ripe79.ripe.net/wp-content/uploads/presentations/42-single-radius-experience-ripe79.pdf).
    - **single_radius_georesolver_ping**: Contains the result of measurements made by GeoResolver on the same set of IP addresses of single_radius_ping. This dataset is used to compare the performance of GeoResolver and single-radius.

- **Traceroute tables**:
    - **vps_meshed_traceroutes**: Traceroutes between each RIPE Atlas probes and its 50 closest RIPE Atlas probes. This dataset is meant to measure the last mile delay between each probes and the first observe public IP address. 
    - **vps_meshed_traceroutes_ipv6**: Same as previous table but for IPv6.

- **VPs tables**:
    - **vps_raw**: All connected RIPE Atlas probes with a valid IPv4 address.
    - **vps_filtered**: Filtered RIPE Atlas probes based on the least speed of light violation using vps_meshed_pings measurement.
    - **vps_filtered_final**: Filtered RIPE Atlas probes from vps_filtered, this time, all probes without a latency under 2ms towards at least one public IP address was removed (using traceroutes table).
    - **vps_filtered_ipv6**: Same as vps_filtered but in IPv6 (Note: due to a bug vps_filtered_final_ipv6 is not available).
    - **vps_filtered_final_CoNEXT_winter_submision**: Same as vps_filtered_final but produced for CoNEXT winter submission.
    
- **DNS tables**:
    - **vps_ecs_mapping**: Latest ECS-DNS resolution over all vantage points' /24 prefixes over the set GeoResolver's selected hostnames.
    - **vps_ecs_mapping__2025_04_13**: Same as vps_ecs_mapping but historic data.
    - **vps_ecs_mapping_ecs_ipv6_latest**: Same as vps_ecs_mapping but for IPv6.
    - **meshed_cdns_ecs**: ECS-DNS resolution over all vantage points specically ran for CDNs evaluation (related with meshed_ping_cdns). 

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
Alternatively, you can execute each script individually. You may notice that some steps within the evaluation scripts are intentionally skipped. This is because these scripts are also designed to run the measurements used during the evaluation phase. **WE STRONGLY DISCOURAGE** executing these measurements, as they impose a significant load on the RIPE Atlas platform. In particular, many of them perform *meshed measurements* — i.e., probes from all vantage points to a set of targets — which is resource-intensive and costly.

Instead, in the following section, we guide you through running your own measurements using GeoResolver. This approach is lightweight and operates with a fixed cost, using 50 vantage points to geolocate each target IP address.

# 🗺️ Run your own experiments

Run GeoResolver script:
```bash
python georesolver/scripts/local_demo.py
```
or
```bash
poetry run python georesolver/scripts/local_demo.py
```

Note: This section requires a [RIPE Atlas account](https://atlas.ripe.net/) with sufficient credits to run measurements. If you don’t already have an account, create one and request credits as needed, RIPE Atlas provides credits to researchers upon request.

The provided Python script selects a random set of 100 ICMP-responsive IP addresses from the ITDK dataset. It then runs the GeoResolver geolocation engine on this set. GeoResolver launches four parallel processes:
1. **ECS process**: Executes ECS-DNS queries on the subnets of all target IP addresses.
2. **Score process**: Computes redirection distances between the ECS resolutions from all vantage points and those of the targets.
3. **Ping process**: Selects the vantage points with the smallest redirection distances and initiates measurements via RIPE Atlas.
4. **Insert process**: Stores the resulting ping measurements into a ClickHouse table.

GeoResolver follows a waterfall architecture: each process operates on batches and triggers the next process upon completion. Execution progress can be monitored via logs located in [georesolver/logs/local_demo/](), where each process maintains a separate log file.

At the end of the pipeline, four tables should be created:
- local_demo_ecs
- local_demo_score
- local_demo_ping
- local_demo_geoloc (containing final geolocation results based on the shotest ping vantage point)

Additionally, you can generate a visualization of the geolocated IP addresses using the script:
```bash
python geresolver/scripts/geoloc_map.py 
```
By default, this script outputs only IP addresses geolocated within a 2ms latency threshold.

**Note 1**: You can reniew vantage points ECS-DNS resolution by setting the following variable to True in the JSON config of [georesolver/scripts/geoloc_map.py](georesolver/scripts/geoloc_map.py):
```python
"init_ecs_mapping": True, # Set to True to renew VPs ECS resolution
```
⚠️ This step takes several hours to complete.

**Note 2**: You can choose to load any target file as input (this is the script we used to run GeoResolver on the ITDK dataset), simply verify that your IP addresses answer to pings! (As mentionned in the paper, GeoResolver methodology works for IPv6 but we did not yet have the tool available for the latter).

**Note 3**: For the same reason we do not recommend to run the meshed measurements of some evaluation, we kindly invite you to use the sanitize vantage points selection available on the FTP. Filtering vantage points also requires *meshed pings* between all vantage points and RIPE Atlas anchors. 


# Acknowledgements

We thank the anonymous reviewers from the ACM CoNEXT program committee for their thoughtful reviews. We thank Maxime Mouchet from IPInfo for sharing their geolocation data via their academic program. Hugo Rimlinger, Olivier Fourmaux, and Timur Friedman are supported by a grant from the French Ministry of Armed Forces.

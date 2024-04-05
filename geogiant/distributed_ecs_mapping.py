from loguru import logger

from fabric import Connection
from geogiant.common.settings import PathSettings

path_settings = PathSettings()

gcp_vms = {
    "iris-asia-east1": "35.206.250.197",
    # "iris-asia-northeast1": "35.213.102.165",
    # "iris-asia-southeast1": "35.213.136.86",
    # "iris-us-east4": "35.212.77.8",
    # "iris-southamerica-east1": "35.215.236.49",
    # "iris-asia-south1": "35.207.223.116",
    # "iris-europe-north1": "35.217.61.50",
    # "iris-europe-west6": "35.216.205.173",
    # "iris-us-west4": "35.219.175.87",
    # "iris-me-central1": "34.1.33.16",
}

c = Connection(f"hugo@{gcp_vms['iris-asia-east1']}")

result = c.run("rm -rf GeoGiant")
result = c.run(
    f"git clone https://{path_settings.GITHUB_TOKEN}@github.com/dioptra-io/GeoGiant.git"
)
result = c.run("ls -l")

result = c.run("cd GeoGiant && poetry lock && poetry install")

result = c.run("mkdir -p GeoGiant/geogiant/datasets")

result = c.run("git clone https://github.com/zmap/zdns.git")

result = c.run("cd zdns && go build")

result = c.run("cp zdns/zdns GeoGiant/geogiant/zdns/zdns_binary")

result = c.put(
    local=f"{path_settings.DEFAULT}/../datasets/ecs_selected_hostnames.csv",
    remote="GeoGiant/geogiant/datasets",
)

result = c.put(
    local=f"{path_settings.DEFAULT}/../datasets/vps_subnet.json",
    remote="GeoGiant/geogiant/datasets",
)

logger.info("Starting hostname init process")
# result = c.run("cd GeoGiant && poetry run python geogiant/hostname_init.py")

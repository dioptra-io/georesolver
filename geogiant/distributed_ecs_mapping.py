from loguru import logger

from fabric import Connection
from geogiant.common.settings import PathSettings

path_settings = PathSettings()

c = Connection("hugo@venus.planet-lab.eu")

# result = c.run("rm -rf GeoGiant")
# result = c.run(
#     f"git clone https://{path_settings.GITHUB_TOKEN}@github.com/dioptra-io/GeoGiant.git"
# )
# result = c.run("ls -l")

# result = c.run("cd GeoGiant && poetry lock && poetry install")

# result = c.run("mkdir -p GeoGiant/geogiant/datasets")

result = c.put(
    local=f"{path_settings.DEFAULT}/../datasets/ecs_selected_hostnames.csv",
    remote="GeoGiant/geogiant/datasets",
)
result = c.put(
    local=f"{path_settings.DEFAULT}/../datasets/vps_subnet.json",
    remote="GeoGiant/geogiant/datasets",
)

logger.info("Starting hostname init process")
result = c.run("cd GeoGiant && poetry run python geogiant/hostname_init.py")

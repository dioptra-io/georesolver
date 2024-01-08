from loguru import logger
from ripe_atlas_module import RIPEAtlas
from common.settings import CommonSettings

if __name__ == "__main__":
    hostname_file = CommonSettings().HOSTNAME_RAW
    name_servers = "8.8.8.8" # TODO: add resolvers
    subnets = ["132.227.78.108/24"]
    output_table = "test_dns_mapping"
    
    subnet_mapping = RIPEAtlas().run()

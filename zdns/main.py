from loguru import logger
from zdns_module import ZDNS
from common.settings import CommonSettings

if __name__ == "__main__":
    hostname_file = CommonSettings().HOSTNAME_RAW
    name_servers = "8.8.8.8" # TODO: add resolvers
    subnets = ["132.227.78.108/24"]
    output_table = "test_dns_mapping"
    
    subnet_mapping = ZDNS(subnets,hostname_file, name_servers, output_table).run()

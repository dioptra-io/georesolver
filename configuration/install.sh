#!/bin/bash

STORAGE_PATH=$1
EXEC_PATH="$CONFIG_PATH/clickhouse"

if [ $# -ne 1 ]; 
    then 
        echo "Script install.sh requires a single storage path as input parameter"
        return 1
fi

# load env var if not global
source .env

# check if clickhouse credentials were correctly set
if [[ -z "$CLICKHOUSE_DATABASE" ]]; 
    then
        echo "CLICKHOUSE credentials env var must be set up"
        return 1
fi


# check if clickhouse client installed
if [ -f $EXEC_PATH ]; then
    echo "Clickhouse client file already downloaded"
else
    echo "Installing Clickhouse from exec file"
    curl https://clickhouse.com/ | shd
    mv clickhouse $CONFIG_PATH
fi

# start clickhouse server
echo "testing Clickhouse server connectivity"
curl http://0.0.0.0:8123

if [ "$?" -eq 0 ]; then 
    echo "A Clickhouse server is already runnning"
else
    echo "Starting Clickhouse server"

    # pull the docker image
    docker pull clickhouse/clickhouse-server:22.6

    docker run -d \
        -p 127.0.0.1:8123:8123 \
        -p 127.0.0.1:9000:9000 \
        -e CLICKHOUSE_MAX_QUERY_SIZE=10000000000000000000 \
        -v $STORAGE_PATH/data:/var/lib/clickhouse/ \
        -v $STORAGE_PATH/logs:/var/log/clickhouse-server/ \
        --ulimit nofile=262144:262144 \
        clickhouse/clickhouse-server:22.6

    sleep 2
    
fi

echo "Creating GeoResolver database and user"
# create project database, user and user grants
$EXEC_PATH client --user $CLICKHOUSE_ADMIN_USERNAME --password $CLICKHOUSE_ADMIN_PASSWORD --query "CREATE DATABASE IF NOT EXISTS $CLICKHOUSE_DATABASE"
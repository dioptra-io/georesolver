#!/bin/bash
CLICKHOUSE_PATH="/storage/clickhouse"
CONFIG_PATH="$PWD/configuration/clickhouse"
EXEC_PATH="$CONFIG_PATH/clickhouse"

# check if clickhouse client installed
if [ -f $EXEC_PATH ]; then
    echo "Clickhouse client file already downloaded"
else
    echo "Installing Clickhouse from exec file"
    curl https://clickhouse.com/ | sh
    mv clickhouse $CONFIG_PATH
fi

# start clickhouse server
echo "testing Clickhouse server connectivity"
curl http://localhost:8123

if [ "$?" -eq 1 ]; then 
    echo "A Clickhouse server is already runnning"
else
    echo "Starting Clickhouse server"

    # pull the docker image
    docker pull clickhouse/clickhouse-server:22.6

    docker run -d \
        -p 127.0.0.1:8123:8123 \
        -p 127.0.0.1:9000:9000 \
        -p 127.0.0.1:9009:9009 \
        -v $CLICKHOUSE_PATH/data:/var/lib/clickhouse/ \
        -v $CLICKHOUSE_PATH/logs:/var/log/clickhouse-server/ \
        -v $PWD/configuration/clickhouse/config.xml:/etc/clickhouse-server/user.xml \
        --ulimit nofile=262144:262144 \
        clickhouse/clickhouse-server:22.6

    sleep 2
    
    # create project database
    $EXEC_PATH client --query "CREATE DATABASE IF NOT EXISTS geogiant"
fi

# install project dependencies with poetry
# poetry lock 
# poetry install

#!/bin/bash

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

if [ "$?" -eq 0 ]; then 
    echo "A Clickhouse server is already runnning"
else
    echo "Starting Clickhouse server"

    # pull the docker image
    docker pull clickhouse/clickhouse-server:22.6

    docker run -d \
        -p 8123:8123 \
        -p 9000:9000 \
        -v /Users/hugo/clickhouse/data:/var/lib/clickhouse/ \
        -v $CONFIG_PATH/logs:/var/log/clickhouse-server/ \
        --ulimit nofile=262144:262144 \
        clickhouse/clickhouse-server:22.6

    sleep 2
    
    # create the database
    $EXEC_PATH client --query "CREATE DATABASE IF NOT EXISTS geogiant"
fi

# install project with poetry
poetry lock 
poetry install
